import numpy as np
import net
from pathlib import Path
import multiprocessing as mp

class BaseInteractor:
    def __init__(self, targets=None, normalizeTargets=False, averagingTime=1):
        '''Initialize Interactor object

        This one returns empty stimulation arrays, but serves as an interface
            for other Interactor classes. It has the base capability to
            optionally  accept a target array at initialization, and score
            output arrays against the target array.

        Arguments:
            targets = a 1D numpy array of target outputs, one per output neuron
            averagingTime = (optional) an amount of time over which to
                average output before scoring. Default is 1 (only one timestep)
            normalizeTargets = (optional) boolean indicating that the targets
                array is not normalized, and should be normalized. Note that
                a copy of the targets array is not made before normalization,
                so this can result in changes in the source array. If this is
                a problem, just supply a copy of the target array, rather than
                the original pointer.
        '''
        self.__t = -1
        self.__history = None
        self.__averagingTime = averagingTime
        if (targets is not None) and normalizeTargets:
            # Normalize target array
            self.__targets = targets / targets.sum()
        else:
            self.__targets = targets

    def next(self, *args, lastOutputs=None, **kwargs):
        '''Return the next set of stimuli for the network.

        lastOutputs: A numpy array of outputs from the net. If provided, the
            interactor may be able to provide closed loop feedback, as well as
            calculate a score for the '''

        if lastOutputs is not None:
            # Update history with new net outputs if given.
            if self.__history is None:
                # Initialize history, since we now know the size of the output
                #   array
                self.__history = np.full([len(lastOutputs), self.__averagingTime], np.nan)
            else:
                # Roll history array to make room for newest output
                self.__history = np.roll(self.__history, 1, axis=1)
            # Set new output
            self.__history[:, 0] = lastOutputs

        self.setTime(self.getTime() + 1)
        return []

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if issubclass(type(other), BaseInteractor):
            return ChainInteractor([self, other])
        elif other == 0:
            # Adding to zero has no effect. This is to allow the use of the
            #   sum function with a list of interactors.
            return self
        else:
            raise ValueError('Can only chain interactors, not objects of type {t}'.format(t=type(other)))

    def __mul__(self, other):
        return StackInteractor([self, other])

    def getTime(self):
        return self.__t

    def setTime(self, t):
        self.__t = t

    def getTargets(self):
        return self.__targets

    def setTargets(self, newTargets):
        deltaTargetSize = len(newTargets) - len(self.__targets)
        if deltaTargetSize > 0:
            # Need to change increase history size
            self.__history = np.concatenate(self.__history, np.full([deltaTargetSize, self.__averagingTime], np.nan))
        elif deltaTargetSize < 0:
            self.__history = self.__history[:deltaTargetSize, :]

        self.__targets = newTargets

    def getHistory(self):
        return self.__history

    def scoreOutput(self):
        '''Return a score representing how far the current average output vector
            is to the target vector, by Euclidean distance.

        Returns:
            a numerical score, where the closer the number to zero, the more
                precisely the average output matches the target vector.'''
        if self.__targets is None:
            raise ValueError('Cannot calculate score because target array has not been provided.')
        if self.__history is None:
            raise ValueError('Cannot calculate score because no outputs have been provided yet.')
        # Average outputs over time
        firingRate = np.nanmean(self.__history, axis=1)
        # Normalize firing rate
        mag = firingRate.sum()
        if mag != 0:
            firingRate = firingRate / firingRate.sum()
        # Calculate output score
        # print('history=   ', self.__history)
        # print('type history=   ', type(self.__history))
        # print('firingRate=', firingRate)
        # print('__targets= ', self.__targets)
        score = np.linalg.norm(firingRate - self.__targets)
        return score

    def restart(self):
        '''Reset Interactor back to initial state'''
        self.__history = None
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
        ''' Initialize TonicInteractor object

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

class NullInteractor(TonicInteractor):
    '''This Interactor provides zero stimulation.'''

    def __init__(self, numInputs, numTimeSteps, *args, **kwargs):
        nullArray = np.zeros(numInputs)
        super().__init__(nullArray, numTimeSteps, *args, **kwargs)

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

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if issubclass(type(other), ChainInteractor):
            # Concatenate the two together
            return ChainInteractor(self.interactors + other.interactors)
        elif other == 0:
            # Adding to zero has no effect. This is to allow the use of the
            #   sum function with a list of interactors.
            return self
        elif issubclass(type(other), BaseInteractor):
            self.interactors.append(other)
            return self
        else:
            raise ValueError('Can only chain interactors, not objects of type {t}'.format(t=type(other)))

    def getTargets(self):
        return self.interactors[self.interactorIdx].getTargets()
    def setTargets(self, newTargets):
        self.interactors[self.interactorIdx].setTargets(newTargets)
    def getHistory(self):
        return self.interactors[self.interactorIdx].getHistory()
    def scoreOutput(self, interactorIdx=None, soFar=False, average=True):
        '''Return the scores for one or more of the chained interactors

        Return a list of output scores, one for each selected interactor, or
          for all of them if interactorIdx is not provied.

        Arguments:
            interactorIdx = a list of interactor indices to select for scoring.
                If None, all interactor scores are returned. Default is None.
            soFar = a boolean flag indicating that only the interactors that
                have run at least one time step should be scored. If soFar is
                false, all of the interactors will be scored, and a ValueError
                will be returned if some of the interactors have not run yet.
                Default is False.
            average = a boolean flag indicating that the output scores should be
                averaged, rather than a list. Default is True
        '''

        scores = []

        # Loop over sub interactors and get score from each
        for idx, interactor in enumerate(self.interactors):
            if interactorIdx is None or idx in interactorIdx:
                # This is a selected interactor.
                if soFar and interactor.history is None:
                    # Only interactors that have run so far are requested, and
                    #   this one hasn't run, so skip scoring it.
                    continue
                # Score the interaction and store the score.
                scores.append(interactor.scoreOutput())

        if average:
            # Return average of the sub interactor scores
            return np.mean(scores)
        else:
            # Return the sub interactor scores separately
            return scores

    def restart(self, *args, **kwargs):
        super().restart(*args, **kwargs)
        self.interactorIdx = 0

class StackInteractor(BaseInteractor):
    '''This interactor will stack stimulation patterns of multiple interactors.

    NOT FULLY IMPLEMENTED YET

    This will allow more complex stimulation patterns. The stimulation input
        each interactor produces will be stacked on top of each other in the
        order the interactors are provided. So, if there are two interactors,
        and one interactor produces an input of [3, 2, 0, 6] and the next
        produces [3, 3, 2], the stacked interactor will produce
        [3, 2, 0, 6, 3, 3, 2]. Note that the interactor will terminate whenever
        the first interactor is complete, even if the other one is not.

    To use this, instantiate multiple other Interactor objects, then pass them
        to the StackInteractor constructor.'''

    def __init__(self, interactors, *args, terminateOnFirst=True, **kwargs):
        ''' Initialize StackInteractor object

        Arguments:
            interactors = a list of Interactor objects to be stacked
            terminateOnFirst = (optional) boolean flag indicating whether or not
                to stop when first subinteractor finishes. If False, the
                stimulation will continue until the last interactor has
                finished. If true, the stimulation will terminate when any one
                interactor has finished. Default is True. NOT IMPLEMENTED YET
        '''
        super().__init__(*args, **kwargs)
        self.interactors = interactors

    def next(self, *args, **kwargs):
        super().next(*args, **kwargs)
        try:
            # Return results of current interactor
            input = []
            for interactor in self.interactors:
                input.extend(interactor.next(*args, **kwargs))
            return input
        except StopIteration:
            # An interactor has completed, terminate
            raise StopIteration('Stacked interactor stimulation has completed.')

    def __mul__(self, other):
        if type(other) == StackInteractor:
            # Stack the two together
            return StackInteractor(self.interactors + other.interactors)
        else:
            if issubclass(type(other), BaseInteractor):
                self.interactors.append(other)
                return self
            else:
                raise ValueError('Can only chain interactors, not objects of type {t}'.format(t=type(other)))

    def getTargets(self):
        return self.interactors[self.interactorIdx].getTargets()
    def setTargets(self, newTargets):
        self.interactors[self.interactorIdx].setTargets(newTargets)
    def getHistory(self):
        return self.interactors[self.interactorIdx].getHistory()
    def scoreOutput(self):
        return self.interactors[self.interactorIdx].scoreOutput()

class FeedbackInteractor(BaseInteractor):
    '''This Interctor will stimulate using a sub-interactor, but will also
        provide feedback to a second set of input neurons proportional to the
        euclidean distance between the net's output and the given target
        pattern.
    '''

    def __init__(self, subInteractor, numFeedbackNeurons, *args,
                    feedbackTimeSteps=None, **kwargs):
        ''' Initialize FeedbackInteractor object

        Arguments:
            subInteractor = an Interactor object that will be used to deliver
                the primary stimulation and score the output
            numFeedbackNeurons = number of neurons to provide error
                feedback to. The feedback will be tacked onto the end of the
                inputs.
            feedbackTimeSteps = (optional) number of timesteps for which to
                provide feedback for. After this time elapses, no feedback
                will be provided. Default is None, which means feedback will
                be provided for the full time specified by the timeSteps
                argument.
        '''
        super().__init__(*args, **kwargs)
        self.subInteractor = subInteractor
        self.numFeedbackNeurons = numFeedbackNeurons
        self.feedbackTimeSteps = feedbackTimeSteps
        self.latestScore = None

    def next(self, *args, **kwargs):
        super().next(*args, **kwargs)

        # Get primary stimulation from subinteractor
        inputs = self.subInteractor.next(*args, **kwargs)
        # Get score from sub interactor
        self.latestScore = self.subInteractor.scoreOutput()
        if self.feedbackTimeSteps is None or self.getTime() < self.feedbackTimeSteps:
            # Construct feedback array from score
            feedback = np.full(self.numFeedbackNeurons, self.latestScore)
        else:
            feedback = np.full(self.numFeedbackNeurons, np.NaN)
        # Add feedback inputs to end of regular imputs
        fullInputs = np.concatenate([inputs, feedback])
        return fullInputs

    def getTargets(self):
        return self.subInteractor.getTargets()
    def setTargets(self, newTargets):
        self.subInteractor.setTargets(newTargets)
    def getHistory(self):
        return self.subInteractor.getHistory()
    def scoreOutput(self):
        if self.latestScore is None:
            # We haven't already scored the last output
            return self.subInteractor.scoreOutput()
        else:
            # We've already scored the last output - just return that score.
            return self.latestScore

class ConnectomeEvolver:
    '''A class that handles the evolution of a population of connectoms'''
    def __init__(self, seedConnectomes, interactors,
            populationSize=100, keepFrac=0.2, inputRegion="I", outputRegion="O",
            keepSeeds=False, meanNumMutations=2,
            stdNumMutations=0.5):
        '''Instantiate a ConnectomeEvolver


        This defines how to start off the initial population, the inputs and
            output definitions for the networks, and the target outputs that
            the nets will be judged against

        Arguments:
            seedConnectomes = a list of one or more connectome CSV files or
                loaded Connectome objects to serve as the progenitors.
            interactors = a list of one or more Interactor objects (any
                subclass of the BaseInteractor class) that will provide the
                stimulation and output scoring for the networks
            populationSize = (optional) the size of each generation. Default is
                100.
            keepFrac = (optional) the fraction of each population to keep.
                Default is 0.1
            keepSeeds = (optional) boolean flag indicating whether or not to
                keep the unmodified seed connectomes in the output population
            meanNumMutations = (optional) the mean number of mutations for each
                generated connectome
            stdNumMutations = (optional) the standard deviation of number of
                mutations for each generated connectome
            inputRegion = A region to be used for input. This region will never
                be mutated. Note that the seed connectomes must have a region
                by this name with enough neurons for the expected input
            outputRegion = A region to be used for output. This region will
                never be mutated. Note that the seed connectomes must have a
                region by this name with enough neurons for the expected output
        '''

        # If any of the seed connectomes are string paths to csv files, load
        #   them as Connectome objects.
        self.seeds = []
        for seedConnectome in seedConnectomes:
            if type(seedConnectome) == type(str()):
                newConnectome = Connectome(seedConnectome)
            elif issubclass(type(seedConnectome), net.Connectome):
                newConnectome = seedConnectome
            else:
                raise ValueError('Seed connectomes must be either strings representing paths to connectome files, or net.Connectome objects')
            self.seeds.append(newConnectome)
        self.population = []
        self.interactors = interactors
        self.populationSize = populationSize
        self.keepFrac = keepFrac
        self.inputRegion = inputRegion
        self.outputRegion = outputRegion
        self.keepSeeds = keepSeeds
        self.meanNumMutations = meanNumMutations
        self.stdNumMutations = stdNumMutations

    def fillPopulation(self, seedConnectomes, N, keepSeeds=False, meanNumMutations=2, stdNumMutations=0.5):
        '''Take a seed populations, and randomly propagate them.

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
        childCount = N-len(newPopulation)
        parents = np.random.choice(seedConnectomes, size=childCount)
        mutationCounts = np.random.normal(loc=meanNumMutations, scale=stdNumMutations, size=childCount).round().astype('int')
        for k, parent in enumerate(parents):
            child = parent.copy()
            for m in range(mutationCounts[k]):
                child.mutate(immutableRegions=[self.inputRegion, self.outputRegion])
            newPopulation.append(child)
        return newPopulation

    def scoreConnectome(self, connectome, nNets, testsPerNet):
        '''Test the given Connectome object and give it a score

        Each connectome will be used to generate nNets nets. The interactors
            loaded into the ConnectomeEvolver object will be used to test and
            evalulate each net.

        Arguments:
            connectome = a Connectome object
            nNets = the number of nets each Connectome object will be used to
                generate. The average score of the generated nets will be
                reported for each Connectome object.
            testsPerNet = number of interactor objects to randomly choose and
                test each generated net on

        Returns:
            A numerical score representing how accurately the nets generated
                from the Connectome produced the target pattern on average.

        '''
        scores = []
        # Generate a list of stim/target pairs, with appropriate randomization,
        #   to test each net on
        for k in range(nNets):
            # Create a chain interactor with all the randomly chosen interactors
            #   chained one after another.
            print('        Creating net #{n} of {nn}'.format(n=k+1, nn=nNets))
            interactor = sum(np.random.choice(self.interactors, size=testsPerNet))
            subScores = []
            n = connectome.createNet()
            output = n.runInteraction(interactor,
                inputAttributeName='region',
                inputAttributeValue=self.inputRegion,
                inputMapped=True,
                outputAttributeName='region',
                outputAttributeValue=self.outputRegion,
                outputMapped=True)
            score = interactor.scoreOutput()
            scores.append(score)
            print('            Net score: {s}'.format(s=score))

        finalScore = np.mean(scores)
        print('    Connectome score = {s}'.format(s=finalScore))
        return finalScore

    def scoreOutput(self, output, target, normalizeTargets=False):
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

    def evolve(self, nGens=10, nNets=10, testsPerNet=3, saveDir='.', saveName='survivor',
        saveAllGenerations=False, numWorkers=None):
        '''Evolve a connectome object

        Arguments:
            nGens = number of generations to evolve
            saveDir = path to a directory in which to put the generation folders
                with the saved connectomes.
            saveName = a string to prepend to the saved survivor connectome
                files
            saveAllGenerations = a boolean flag indicating whether to save
                every connectome, or only the last generation ones.
            keepSeeds = (optional) boolean flag indicating whether or not to
                keep the unmodified seed connectomes in the output population
            meanNumMutations = (optional) the mean number of mutations for each
                generated connectome
            stdNumMutations = (optional) the standard deviation of number of
                mutations for each generated connectome
            numWorkers = (optional) number of parallel processes to use to
                evaluate connectomes. Default is None, which means the
                connectomes will be evaluated serially in a single process.
        '''

        self.population = self.seeds
        for g in range(nGens):
            print('Running generation #{g} of {gn}'.format(g=g, gn=nGens))
            self.population = self.fillPopulation(self.population, self.populationSize, keepSeeds=True, meanNumMutations=2, stdNumMutations=0.5)
            scores = []
            if numWorkers is None:
                for k, co in enumerate(self.population):
                    print('    Testing connectome #{k} of {kn}'.format(k=k+1, kn=len(self.population)))
                    scores.append(self.scoreConnectome(co, nNets=nNets, testsPerNet=testsPerNet))
            else:
                with mp.Pool(processes=numWorkers) as pool:
                    results = []
                    for k, co in enumerate(self.population):
                        print('    Testing connectome #{k} of {kn}'.format(k=k+1, kn=len(self.population)))
                        result = pool.apply_async(self.scoreConnectome, (co,), dict(nNets=nNets, testsPerNet=testsPerNet))
                        results.append(result)
                    print('    Waiting for results...')
                    scores = [result.get() for result in results]
            # Sort population and scores
            sortedPopulation, sortedScores = zip(*sorted(zip(self.population, scores), key=lambda p:p[1]))
            numToKeep = round(self.keepFrac * len(self.population))
            survivors = sortedPopulation[0:numToKeep]
            survivorScores = sortedScores[0:numToKeep]
            if saveAllGenerations or g == nGens-1:
                print('Saving {n} survivors.'.format(n=numToKeep))
                for k in range(numToKeep):
                    savePath = Path(saveDir) / '{n}_gen{g:03d}_score{s:.2e}.csv'.format(n=saveName, g=g, s=survivorScores[k])
                    print('Saving survivor #{k}'.format(k=k))
                    survivors[k].save(savePath)
            self.population = survivors
            print('Survivor scores:')
            print(survivorScores)

if __name__ == "__main__":
    stim1 = np.array([[1, 2, 3, 4], [11, 22, 33, 44], [111, 222, 333, 444]]);
    targets = np.array([-1, -3, 0, -1, 4])
    pi = PredefinedInteractor(stim1, targets=targets, normalizeTargets=True)
    stim2 = np.array([7, 7, 1, 0])
    ti = TonicInteractor(stim2, 10, targets=targets, normalizeTargets=True, averagingTime=5)
    stim3 = np.array([123, 234, 345, 456])
    pat3 = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    pati = PatternInteractor(stim3, pat3, targets=targets, normalizeTargets=True)
    chi = pi + ti + pati
    ni = NullInteractor(5, 15, targets=[0, 0, 0, 0, 0])
    fbi = FeedbackInteractor(ti, 3)
    iact = fbi + fbi + fbi
    while True:
        try:
            print('Stim:', iact.next(lastOutputs=np.random.rand(5)))
            print('History:', iact.getHistory())
            print('Score:', iact.scoreOutput())
        except StopIteration:
            break;
