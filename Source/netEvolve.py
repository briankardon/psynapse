import numpy as np

class BaseInteractor:
    def __init__(self):
        '''Initialize Interactor object

        This one returns empty stimulation arrays, but serves as an interface
            for other Interactor classes.
        '''
        self.__t = -1

    def next(self, *args, lastOutput=None, **kwargs):
        self.setTime(self.getTime() + 1)
        return []

    def getTime(self):
        return self.__t

    def setTime(self, t):
        self.__t = t

    def restart(self):
        '''Reset Interactor back to initial state'''
        self.setTime(-1)

class PredefinedInteractor(BaseInteractor):
    '''An Interactor class that stimulates using a predefined array of inputs,
        one per neuron per time point.
    '''
    def __init__(self, stimPattern, *args, **kwargs):
        '''Initialize Interactor object'''
        super().__init__(*args, **kwargs)
        self.stimPattern = stimPattern

    def next(self, *args, **kwargs):
        super().next(*args, **kwargs)
        try:
            return self.stimPattern[:, self.getTime()]
        except IndexError:
            raise StopIteration('End of stimulation pattern')

class TonicInteractor(BaseInteractor):
    '''This Interactor will just stimulate with the given input
        over and over for the given number of time steps.
    '''

    def __init__(self, inputs, timeSteps, *args, **kwargs):
        ''' Initialize RepeatInteractor object

        Arguments:
            inputs = a 1D array of inputs, one per input neuron
            timeSteps = an integer indicating how many timesteps to stimulate
                for
        '''
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.timeSteps = timeSteps

    def next(self, *args, **kwargs):
        super().next(*args, **kwargs)
        if self.getTime() < self.timeSteps:
            return self.inputs
        else:
            raise StopIteration('End of stimulation pattern')

class PatternInteractor(BaseInteractor):
    '''This Interactor will stimulate with the given input array following a
        given temporal pattern, such that the temporal and neuronal stimulation
        is separable.
    '''

    def __init__(self, inputs, pattern, *args, **kwargs):
        ''' Initialize PatternInteractor object

        Arguments:
            inputs = a 1D numpy array of inputs, one per input neuron
            pattern = a 1D iterable of booleans indicating whether to
                stimulate with the given input pattern, or not stimulate for
                each time step
        '''
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.null = 0*self.inputs
        self.pattern = pattern

    def next(self, *args, **kwargs):
        super().next(*args, **kwargs)
        try:
            if self.pattern[self.getTime()]:
                return self.inputs
            else:
                return self.null
        except IndexError:
            raise StopIteration('End of stimulation pattern')

class ChainInteractor(BaseInteractor):
    '''This interactor will chain multiple interactors together.

    This will allow multiple interactions to be applied, each one after the
        last has ended, but using the same interface as a single interactor.

    To use this, instantiate multiple other Interactor objects, then pass them
        to the ChainInteractor constructor.'''

    def __init__(self, interactors, *args, **kwargs):
        ''' Initialize ChainInteractor object

        Arguments:
            interactors = a list of Interactor objects to be chained together
        '''
        super().__init__(*args, **kwargs)
        self.interactors = interactors
        self.interactorIdx = 0

    def next(self, *args, **kwargs):
        super().next(*args, **kwargs)
        try:
            # Return results of current interactor
            return self.interactors[self.interactorIdx].next(*args, **kwargs)
        except StopIteration:
            # This interactor is complete, move on to next one
            self.interactorIdx += 1
            return self.next(*args, **kwargs)
        except IndexError:
            # Ran out of interactors
            raise StopIteration('Final interactor in chain is complete.')

class FeedbackInteractor(BaseInteractor):
    '''This Interctor will stimulate with the given input over and over for the
        given number of time steps, but will provide feedback to a second set
        of input neurons proportional to the euclidean distance between the
        net's output and the target pattern.
    '''

    def __init__(self, inputs, targets, numFeedbackNeurons, timeSteps, *args,
                    feedbackAveragingTime=1, feedbackTimeSteps=None, **kwargs):
        ''' Initialize FeedbackInteractor object

        Arguments:
            inputs = a 1D numpy array of inputs, one per input neuron
            targets = a 1D numpy array of target outputs, one per output neuron
            numFeedbackNeurons = number of neurons to provide error
                feedback to. The feedback will be tacked onto the end of the
                inputs.
            timeSteps = an integer indicating how many timesteps to stimulate
                for
            feedbackAveragingTime = (optional) an amount of time over which to
                average output before scoring. Default is 1 (only one timestep)
            feedbackTimeSteps = (optional) number of timesteps for which to
                provide feedback for. After this time elapses, no feedback
                will be provided. Default is None, which means feedback will
                be provided for the full time specified by the timeSteps
                argument.
        '''
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.targets = targets / targets.sum()
        self.numFeedbackNeurons = numFeedbackNeurons
        self.timeSteps = timeSteps
        self.feedbackAveragingTime = feedbackAveragingTime
        if feedbackTimeSteps is None:
            # User did not specify amount of time to provide feedback, so we
            #   will provide feedback for the full time.
            self.feedbackTimeSteps = self.timeSteps
        else:
            self.feedbackTimeSteps = feedbackTimeSteps
        # We will dynamically initialize history, because we don't know how
        #   many output neurons there will be.
        self.history = None

    def next(self, outputs, *args, **kwargs):
        super().next(*args, **kwargs)

        if self.history is None:
            # Initialize history
            self.history = np.full([len(outputs), self.feedbackAveragingTime], np.nan)

        # Roll history array to make room for newest output
        np.roll(self.history, 1, axis=1)
        # Set new output
        self.history[:, 0] = output
        # Average outputs over time
        firingRate = np.nanmean(self.history, axis=1)
        # Normalize firing rate
        firingRate = firingRate / firingRate.sum()
        # Calculate output score
        score = self.scoreOutput(firingRate, targets)

        if self.getTime() < self.timeSteps:
            feedback = np.full(self.numFeedbackNeurons, score)
            fullInputs = np.concatenate(self.inputs, feedback)
            return self.inputs
        else:
            raise StopIteration('End of stimulation pattern')

    def scoreOutput(self, outputFiringRates, targetFiringRates):
        score = np.linalg.norm(outputFiringRates - targetFiringRates)

class ConnectomeEvolver:
    '''A class that handles the evolution of a population of connectoms'''
    def __init__(self, seedConnectomes, stimPatterns, targetPatterns,
            randomizer=None, populationSize=100, keepFrac=0.2,
            inputIndices=None, inputAttributeName=None, inputAttributeValue=None,
            outputIndices=None, outputAttributeName=None, outputAttributeValue=None,
            keepSeeds=False, meanNumMutations=2, stdNumMutations=0.5):
        '''Instantiate a ConnectomeEvolver

        This defines how to start off the initial population, the inputs and
            output definitions for the networks, and the target outputs that
            the nets will be judged against

        Arguments:
            seedConnectomes = a list of one or more connectome CSV files or
                loaded Connectome objects to serve as the progenitors.
            stimPatterns = a list of one or more numpy arrays representing
                stimulation patterns to apply to the network. Each array must
                be numerical of size NxT, where N is the number of input
                neurons, and T is the # of time steps over which the net will
                be simulated.
            targetPatterns = a list of one or more numpy arrays representing
                the target pattern that net outputs will be judged against.
                Each array should be MxTo, where M is the number of output
                neurons, and To is the number of time steps over which the net
                will be evaluated. It must be that To <= T
            randomizer = (optional) a function that takes one stimPattern and
                one targetPattern, and returns a randomized stimPattern and
                targetPattern - for input augmentation purposes. Default is
                0.1.
            populationSize = (optional) the size of each generation. Default is
                100.
            keepFrac = (optional) the fraction of each population to keep.
                Default is 0.1
            inputIndices = (optional) either an iterable or slice object
                indicating indices of neurons to set stimulation activation for
            inputAttributeName = (optional) the attribute name corresponding to
                the attribute values given
            inputAttributeValue = (optional) the attribute value to use to
                select neurons to set stimulation activations for
            outputIndices = (optional) either an iterable or slice object
                indicating indices of neurons to get output from
            outputAttributeName = (optional) the attribute name corresponding to
                the attribute values given
            outputAttributeValue = (optional) the attribute value to use to
                select neurons to get output from
            keepSeeds = (optional) boolean flag indicating whether or not to
                keep the unmodified seed connectomes in the output population
            meanNumMutations = (optional) the mean number of mutations for each
                generated connectome
            stdNumMutations = (optional) the standard deviation of number of
                mutations for each generated connectome
        '''

        # If any of the seed connectomes are string paths to csv files, load
        #   them as Connectome objects.
        for k in range(seedConnectomes):
            if type(seedConnectomes[k]) == type(str()):
                newConnectome = Connectome(seedConnectomes[k])
            else:
                newConnectome = seedConnectomes[k]
                self.seeds.append()
                self.population.append()

        self.stimTargetPairs = zip(stimPatterns, targetPatterns)
        if randomizer is None:
            # No randomizer given, make an identity "randomizer"
            self.randomizer = lambda stimPattern, targetPattern: (stimPattern, targetPattern)
        else:
            self.randomizer = randomizer
        self.populationSize = populationSize
        self.keepFrac = keepFrac
        self.inputIndices = inputIndices
        self.inputAttributeName = inputAttributeName
        self.inputAttributeValue = inputAttributeValue
        self.outputIndices = outputIndices
        self.outputAttributeName = outputAttributeName
        self.outputAttributeValue = outputAttributeValue
        self.keepSeeds = keepSeeds
        self.meanNumMutations = meanNumMutations
        self.stdNumMutations = stdNumMutations

    def fillPopulation(self, seedConnectomes, N, keepSeeds=False, meanNumMutations=2, stdNumMutations=0.5):
        '''Take a set of seed populations, and randomly propagate them.

        Arguments:
            seedConnectomes = either a list of connectome files, or a list of
                loaded Connectome objects, to seed the population
            N = desired size of population after propagation
            keepSeeds = (optional) boolean flag indicating whether or not to
                keep the unmodified seed connectomes in the output population
            meanNumMutations = (optional) the mean number of mutations for each
                generated connectome
            stdNumMutations = (optional) the standard deviation of number of
                mutations for each generated connectome

        Returns:
            A list of Connectome objects representing a randomly generated
                population.
        '''

        newPopulation = []
        if keepSeeds:
            newPopulation = [c.copy() for c in seedConnectomes]
        else:
            newPopulation = []
        childCount = N-length(newPopulation)
        parents = np.random.choice(seedConnectomes, size=childCount)
        mutationCounts = np.random.normal(loc=meanNumMutations, scale=stdNumMutations, size=childCount)
        for k, parent in enumerate(parents):
            child = parent.copy()
            for m in range(mutationCounts[k]):
                child.mutate()
            newPopulation.append(child)
        return newPopulation

    def scoreConnectome(self, connectome, nNets, testsPerNet):
        '''Test the given Connectome object and give it a score

        Each connectome will be used to generate nNets nets. The stimulation and
            test patterns loaded into the ConnectomeEvolver object will be used
            to test and evalulate each net.

        Arguments:
            connectome = a Connectome object
            nNets = the number of nets each Connectome object will be used to
                generate. The average score of the generated nets will be
                reported for each Connectome object.
            testsPerNet = number of stim/target pairs to randomly choose and
                test each generated net on

        Returns:
            A numerical score representing how accurately the nets generated
                from the Connectome produced the target pattern on average.

        '''
        scores = []
        # Generate a list of stim/target pairs, with appropriate randomization,
        #   to test each net on
        for k in range(nNets):
            stimTargetPairs = [self.randomizer(stim, target) for stim, target in np.random.choice(self.stimTargetPairs, size=testsPerNet)]
            for stim, target in stimTargetPairs:
                subScores = []
                n = connectome.createNet()
                output = n.runSequence(self.stimPattern,
                    inputIndices=self.inputIndices, inputAttributeName=self.inputAttributeName,  inputAttributeValue=self.inputAttributeValue,
                    outputIndices=self.outputIndices, outputAttributeName=self.outputAttributeName, outputAttributeValue=self.outputAttributeValue)
                score = self.scoreOutput(output, target)
                scores.append(score)
        return np.mean(scores)

    def scoreOutput(self, output, target, normalizeTarget=False):
        '''Score an output matrix by comparing it to a target output matrix

        Arguments:
            output: NxT array, where N is the number of output neurons, and
                T is the number of time steps in the output
            target: NxT array of ideal outputs. Must be the same size as
                the output array.

        '''

        # Calculate an activity vector, giving the mean activity for each neuron
        targetFiringRates = target.mean(axis=1)
        outputFiringRates = output.mean(axis=1)

        # Normalize activity vector
        outputFiringRates /= outputFiringRates.sum()

        nTarget = targetFiringRates.shape[0]
        nOutput = outputFiringRates.shape[0]
        if nTarget > nOutput:
            # If there are too few outputs, it's a fail
            # This check may not be necessary
            return np.Inf

        # Calculate the euclidean distance between the output and target
        #   activity vectors
        score = np.linalg.norm(outputFiringRates - targetFiringRates)

        return score

    def evolve(self, nGens, saveDir, saveAllGenerations=False):
        '''Evolve a connectome object

        Arguments:
            nGens = number of generations to evolve
            saveDir = path to a directory in which to put the generation folders
                with the saved connectomes.
            saveAllGenerations = a boolean flag indicating whether to save
                every connectome, or only the last generation ones.
            keepSeeds = (optional) boolean flag indicating whether or not to
                keep the unmodified seed connectomes in the output population
            meanNumMutations = (optional) the mean number of mutations for each
                generated connectome
            stdNumMutations = (optional) the standard deviation of number of
                mutations for each generated connectome
        '''

        for g in range(nGens):
            print('Running generation #{g}'.format(g=g))
            self.population = self.fillPopulation(self.seeds, self.populationSize, keepSeeds=True, meanNumMutations=2, stdNumMutations=0.5)
            scores = []
            for c in self.population:
                scores.append(self.scoreConnectome(co))
            # Sort population and scores
            sortedPopulation, sortedScores = zip(*[p[0] for p in sorted(zip(scores, self.population), key=lambda p:p[0])])
            numToKeep = round(self.keepFrac * len(self.population))
            survivors = sortedPopulation[0:numToKeep]
            survivorScores = sortedScores[0:numToKeep]
            self.population = survivors
            print('Survivor scores:')
            print(survivorScores)
