import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import csv
import sys
import re

class Net:
    ROTATION = np.array([[0, -1], [1, 0]])
    def __init__(self, numNeurons=1, refractoryPeriods=None,
        refractoryPeriodMean=4, refractoryPeriodSigma=3, thresholds=None,
        thresholdMean=1, thresholdSigma=0, historyLength=700,
        hebbianPlasticityRate=0.1, homeostaticPlasticityFactor=0.1):
        """A class representing a simulated neural network.

        Neurons are are initially unconnected. Apply a connection algorithm
        to connect the neurons.

        Arguments:
            numNeurons = # of neurons to create within the net
            refractoryPeriods = an array of refractory periods to assign to
                the neurons. Must be length of numNeurons. If omitted, instead
                the refractory periods will be randomized according to the
                refractoryPeriodMean and refractoryPeriodSigma arguments
            refractoryPeriodMean = the mean random refractory period to give
                the neurons.
            refractoryPeriodSigma = the standard deviation of the random
                refractory period to give the neurons
            thresholds = an array of thresholds to assign to
                the neurons. Must be length of numNeurons. If omitted, instead
                the thresholds will be randomized according to the
                thresholdsMean and thresholdsSigma arguments
            thresholdsMean = the mean random threshold to give the neurons.
            thresholdsSigma = the standard deviation of the random threshold to
                give the neurons
            historyLength = the amount of simulated time to save recorded
                firing patterns
            hebbianPlasticityRate = the rate at which connection strengths
                change for each coincident firing. For example, any connections
                between coincident firing neurons (one then the other) are
                additively increased by this value. Should be small, but greater
                than zero for biologically reasonable behavior. Default is 0.1.
            homeostaticPlasticityFactor = the factor by which connection strengths
                return back to their base values. Every time step, all neuron
                connections are brought closer to their base connection strength
                by an amount that is proportional to the current distance
                between their current strength and their base strength. The
                constant of proportionality is this factor. Default is 0.1.
                Should be between 0 and 1 for biologically reasonable behavior.
          """

        if historyLength < 2:
            raise ValueError('historyLength must be at least 2, to allow for Hebbian learning.')
        self.numNeurons = numNeurons
        self.connections = np.zeros([self.numNeurons, self.numNeurons])   # Direct excitatory/inhibitory connection matrix
        self.baseConnections = self.connections.copy()
        self.modConnections = np.zeros([self.numNeurons, self.numNeurons])  # Modulatory connection matrix
        self.baseModConnections = self.modConnections.copy()
        self.neurons = [Neuron(self, k) for k in range(self.numNeurons)]
        self.activations = np.zeros(self.numNeurons)
        self.history = np.zeros([self.numNeurons, historyLength])
        self.hebbianPlasticityRate = hebbianPlasticityRate
        self.homeostaticPlasticityFactor = homeostaticPlasticityFactor
        if thresholds is None:
            self.baseThresholds = np.random.normal(loc=thresholdMean, scale=thresholdSigma, size=self.numNeurons)
        else:
            try:
                # If it's a numpy array, copy it to ensure separate memory space
                self.baseThresholds = thresholds.copy()
            except AttributeError:
                # Maybe it's just a list?
                self.baseThresholds = np.array(thresholds)
        self.thresholds = self.baseThresholds.copy()
        if refractoryPeriods is None:
            self.refractoryPeriods = np.round(np.random.normal(loc=refractoryPeriodMean, scale=refractoryPeriodSigma, size=self.numNeurons))
        else:
            try:
                # If it's a numpy array, copy it to ensure separate memory space
                self.refractoryPeriods = refractoryPeriods.copy()
            except AttributeError:
                # Maybe it's just a list?
                self.refractoryPeriods = np.array(refractoryPeriods)
        self.refractoryCountdowns = np.zeros(self.numNeurons)
        self.connectionArrows = [[None for k in range(self.numNeurons)] for j in range(self.numNeurons)]
        self.neuronMarkers = [None for k in range(self.numNeurons)]
        # Neurons can be given custom attributes. Each attribute has an entry
        #   in the attributeNames array, and a corresponding row in the
        #   attribute values matrix. Attribute values must be numerical.
        self.attributeNames = []
        self.attributeMaps = []
        self.attributeMapsReversed = []
        self.attributeValues = np.zeros([0, self.numNeurons])

    def saveNet(self, filename):
        """Save a net to a file so it can be loaded later.

        Not implemented yet.
        """
        pass

    def loadNet(self, filename):
        """Load a saved net to a file.

        Not implemented yet.
        """
        pass

    def setHistoryLength(self, newLength):
        oldLength = self.getHistoryLength()
        if newLength > oldLength:
            self.history = np.append(self.history, np.zeros([self.numNeurons, newLength-oldLength]))
        elif oldLength > newLength:
            self.history = self.history[:, :newLength]

    def getHistoryLength(self):
        return self.history.shape[1]

    def addAttribute(self, name, indices=None, initialValues=None, attributeMap=None):
        """Add an attribute to this net, allowing categorization of neurons.

        Each neuron in the net can take any numerical value for each added
            attribute. An optional attributeMap can be supplied to translate
            numerical values into more eaningful values (for example human-
            readable strings)

        Arguments:
            name = name of new attribute of this net (string)
            initialValues = list of initial values, or None to initialize with all
              zeros. If a list, it must have length equal to numNeurons
            attributeMap = a dictionary mapping attribute values to some kind of human readable name
              or None, indicating there is no mapping.
          """

        if name in self.attributeNames:
            KeyError('Attribute "{n}" already exists.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
        self.attributeNames.append(name)
        self.attributeMaps.append(attributeMap)
        if attributeMap is None:
            reverseMap = None
        else:
            reverseMap = dict([(attributeMap[v], v) for v in attributeMap])
            if len(reverseMap) < len(attributeMap):
                warning('Provided attributeMap is not a 1:1 reversible mapping.')
        self.attributeMapsReversed.append(reverseMap)
        self.attributeValues = np.pad(self.attributeValues, ([0, 1], [0, 0]))
        if initialValues is not None:
            self.attributeValues[-1, indices] = initialValues
        # Return the index of the new attribute
        return self.attributeValues.shape[0] - 1

    def removeAttributes(self, name):
        """Delete an attribute from this net."""

        try:
            idx = self.attributeNames.index(name)
        except ValueError():
            raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        self.attributeNames.pop(idx)
        self.attributeMaps.pop(idx)
        self.attributeMapsReversed.pop(idx)
        np.delete(self.attributeValues, idx, 0)

    def getAttributes(self, name, indices=None, mapped=False):
        """ Return an array of values corresponding to the selected attribute

        Optionally get the attribute values for only neurons selected by the
            indices argument. The values can also be optionally returned after
            being mapped by the attributeMap for this attribute.

        Arguments:
            name = name of an attribute of this net (string)
            indices = (optional) list of neuron indices. Only attribute values
                from the specified neurons will be returned. Default behavior
                is for all neurons will be returned.
            mapped = (optional) boolean flag indicating whether to return
                the raw numerical attribute values (False) or to first map the
                attribute values using the supplied attributeMap (True). Default
                is false (return raw numerical values)

        Returns:
            An iterable containing attribute values. If unmapped, a numpy array.
                If mapped, a list.
        """

        try:
            idx = self.attributeNames.index(name)
        except ValueError():
            raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
        if mapped:
            if self.attributeMaps[idx] is None:
                raise KeyError('Cannot return mapped attribute values, because the selected attribute does not have a map.')
            else:
                return [self.attributeMaps[idx][v] for v in self.attributeValues[idx, indices]]
        else:
            return self.attributeValues[idx, indices]

    def filterByAttribute(self, name, value, mapped=False):
        ''' Get a list of neuron indices that match the given attribute value.

        Arguments:
            name = name of an attribute of this net (string)
            value = the attribute value to match when filtering. If mapped
                is False, value should be a raw numerical attribute value.
                If mapped is True, value should be whatever type the
                attributeMap translates into. Value can also be a function,
                in which case it should take a numpy array of values and return
                a numpy array of logicals indicating which neurons should be
                selected
            mapped = (optional) a boolean flag indicating whether to use raw
                or mapped values to filter.

        Returns:
            A filtered list of neuron indices for neurons whose attribute value
                matched the value provided.
        '''
        values = self.getAttributes(name, mapped=mapped)
        try:
            idx = value(values)
        except TypeError:
            if mapped:
                idx = np.arange(self.numNeurons)[[v==value for v in values]]
            else:
                idx = np.arange(self.numNeurons)[np.equal(values, value)]
        return idx

    def getUniqueAttributes(self, name, mapped=False):
        '''Get a unique list of attribute values for the attribute name.

        Arguments:
            name = name of an attribute of this net (string)
            mapped = (optional) a boolean flag indicating whether to return
                unique raw or mapped values. Note that if the attributeMap is
                not 1:1 reversible, this function could return different
                results for mapped or unmapped.

        Returns:
            A list of unique attribute values for the given attribute name,
                either raw numerical values, or mapped values.
        '''
        return np.unique(self.getAttributes(name, mapped=mapped))

    def setAttributes(self, name, values=0, mapped=False, indices=None, addIfNonexistent=True, attributeMap=None):
        '''Set an array of values for the specified attribute

        Arguments:
            name = name of an existing or new attribute of this net (string)
            values = (optional) an attribute value or array of values to set
            mapped = (optional) a boolean flag indicating whether the value
                is a raw numerical value (False) or a mapped value (True).
                Default is False.
            indices = (optional) a list of indexes of neurons to set attribute
                values for
            addIfNonexistent = (optional) a boolean flag indicating whether or
                not to create the attribute if it doesn't exist
            attributeMap = (optional) a map between attribute values and value
                names, only used if we're creating a new attribute
        '''

        try:
            idx = self.attributeNames.index(name)
        except ValueError:
            if addIfNonexistent:
                idx = self.addAttribute(name, attributeMap=attributeMap)
            else:
                raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
        if mapped:
            # Value supplied is a mapped value. Reverse-map value, then set it.
            values = [self.attributeMapsReversed[value] for value in values]
        self.attributeValues[idx, indices] = values

    def activate(self):
        '''Simulate activation of neurons and transmission of action potentials'''

        self.history = np.roll(self.history, 1, axis=1)
        # Determine which neurons will fire
        firing = (self.refractoryCountdowns <= 0) & (self.activations > self.thresholds)
        notFiring =  np.logical_not(firing)
        # Add firing to history
        self.history[:, 0] = firing
        # Pass signal from firing neurons downstream
        self.activations = firing.dot(self.connections)
        # Reset fired neurons to max countdown, continuing decrementing the rest.
        self.refractoryCountdowns = (self.refractoryPeriods * firing) + ((self.refractoryCountdowns - 1) * notFiring)
        # Pass modulatory signals. Add the modulatory signal to the downstream
        #   neuron's threshold, then move all threshold towards each neuron's
        #   base threshold.
        self.thresholds = self.thresholds + firing.dot(self.modConnections) + 0.25*(self.baseThresholds - self.thresholds)
        # Create matrix of firing coincidences (where C[a, b] = 1 if b fired,
        #   and a fired on the last step), then multiply by the hebbian
        #   plasticity factor, to determine learning changes in network
        hebbianPlasticity = np.outer(self.history[:, 1], firing) * self.hebbianPlasticityRate
        # Create matrix to represent homeostatic relaxation of connection strengths
        homeostaticPlasticity = (self.baseConnections - self.connections) * self.homeostaticPlasticityFactor
        self.connections += hebbianPlasticity + homeostaticPlasticity

    def getIndices(self, indices=None, attributeName=None, attributeValue=None):
        '''Get a list of selected neuron indices.

        Arguments:
            indices = (optional) directly return the indices.
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the values with which to select neurons

        Returns:
            A list of indices selected, or if neither a list of indices or an
                attribute name/value pair are supplied, a slice object selecting
                all indices is returned.
        '''

        if attributeName is not None:
            indices = self.filterByAttribute(attributeName, attributeValue)
        if indices is None:
            indices = np.arange(self.numNeurons)
        try:
            # Check if indices is an array or scalar value.
            iter(indices)
        except TypeError:
            # Indices is a scalar value. Turn it into an array.
            indices = np.array(indices)

        return indices

    def getOutput(self, timeLag=0, indices=None, attributeName=None, attributeValue=None):
        '''Get activations of neurons indicated by indices at the indicated time

        Note that this will only get activations that are in the history; i.e.
            manually set activations will not return from this function until
            activate().

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Arguments:
            timeLag = (optional) integer indicating how many time steps back
                to look in time. Default is 0 (most recent activations). Default
                is 0.
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to get output from
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the attribute values to use to select
                neurons to get output from

        Returns:
            The activations of the selected neurons.
          '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        return self.history[indices, timeLag]

    def getOutputSequence(self, times, indices=None, attributeName=None, attributeValue=None):
        '''Get activations of neurons indicated by indices at given timepoints

        Note that this will only get activations that are in the history; i.e.
            manually set activations will not return from this function until
            activate().

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Arguments:
            times = sequence of time points to retrieve outputs from. 0 is the
                most recent history, 1 is from one time step back, and so forth.
                May be any numerical iterable, or a numpy slice object.
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to get output from
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the attribute values to use to select
                neurons to get output from

        Returns:
            The activations of the selected neurons.
        '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        return self.history[indices, times]

    def getFiringRate(self, averagingTime=None, indices=None,
        attributeName=None, attributeValue=None):
        '''Get time-averaged firing rate of neurons indicated by indices

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Arguments:
            averagingTime = (optional) integer indicating how many time steps
                back to look in time. Default is None, which means the entire
                available history (set by the Net.historyLength property) is
                used.
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to get output from
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the attribute values to use to select
                neurons to get output from

        Returns:
            The average firing rate of the selected neurons.
          '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        if averagingTime is None:
            averagingTime = self.getHistoryLength()
        return np.mean(self.history[indices, :averagingTime], axis=1)

    def setInput(self, inputs, indices=None, attributeName=None, attributeValue=None):
        '''Set activations of neurons indicated by indices to given input values

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Arguments:
            inputs = an iterables of less than or equal length to numNeurons
                indicating how much to add to the neuron's activation input
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to set activations for
            attributeName = (optional) the attribute name corresponding to the
                attribute values given
            attributeValue = (optional) the attribute value to use to select
                neurons to set activations for
        '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        self.activations[indices] = inputs

    def addInput(self, inputs, indices=None, attributeName=None, attributeValue=None):
        '''Add input values to activations of neurons indicated by indices

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Note that this is different from "setInput", because it adds to the
            activations, rather than sets them.

        Arguments:
            inputs = an iterables of less than or equal length to numNeurons
                indicating how much to add to the neuron's activation input
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to set activations for
            attributeName = (optional) the attribute name corresponding to the
                attribute values given
            attributeValue = (optional) the attribute value to use to select
                neurons to set activations for
        '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        self.activations[indices] += inputs

    def randomizeConnections(self, n, mu, sigma, indicesA=None, indicesB=None,
            attributeName=None, attributeNameA=None, attributeValueA=None,
            attributeNameB=None, attributeValueB=None, modulatory=False,
            setBase=True, setCurrent=True):
        '''Change random connection strengths in net

        Three options are given to select upstream and downstream neurons to
            make connections between. For upstream neurons, either indicesA may
            be used to directly specify neuron indices, or attributeNameA/
            attributeValueA can be used to filter neurons. For downstream
            neurons, either indicesB or attributeNameB/attributeValueB can be
            used to filter neurons. Additionally, attributeName may be used
            for convenience to specify both upstream and downstream attribute
            name to use for neuron selection.

        Arguments:
            n = number of connections to change
            mu = mean connection strength
            sigma = standard deviation of connection strength
            indicesA = (optional) the indices of neurons to choose from to
                randomly connect from. If None, all neurons may be chosen to
                connect from.
            indicesB = (optional) the indices of neurons to choose from to
                randomly connect to. If None, all neurons may be chosen to
                connect to.
            attributeName = (optional) the attribute name to select both
                upstream and downstream neurons.
            attributeNameA = (optional) the attribute used to select the
                upstream neurons. If indicesA or attributeName is supplied, this
                is ignored.
            attributeValueA = (optional) the attribute value to select upstream
                neurons to connect from.
            attributeNameB = (optional) the attribute used to select the
                downstream neurons. If indicesB or attributeName is supplied,
                this is ignored.
            attributeValueA = (optional) the attribute value to select
                downstream neurons to connect from.
            modulatory = (optional) boolean flag indicating that the modulatory
                network instead of the direct network should be randomized.
                Default is false (direct, not modulatory)
            setBase = (optional) boolean flag indicating whether to change the
                base connection strengths. Default is True.
            setCurrent = (optional) boolean flag indicating whether to change
                the current connection strengths. Default is True.
        '''

        if attributeName is not None:
            attributeNameA = attributeName
            attributenameB = attributeName
        if attributeNameB is None and attributeNameA is not None:
            attributeNameB = attributeNameA
        if attributeValueB is None:
            attributeValueB = attributeValueA
        if indicesB is None:
            indicesB = indicesA
        indicesA = self.getIndices(indices=indicesA, attributeName=attributeNameA, attributeValue=attributeValueA)
        indicesB = self.getIndices(indices=indicesB, attributeName=attributeNameB, attributeValue=attributeValueB)

        try:
            if len(indicesA) == 0:
                # Upstream neuron group is empty do nothing.
                return
        except TypeError:
            # Probably a slice object rather than a list of indices, carry on.
            pass
        try:
            if len(indicesB) == 0:
                # Upstream neuron group is empty do nothing.
                return
        except TypeError:
            # Probably a slice object rather than a list of indices, carry on.
            pass

        # Get random neuron coordinates
        x = np.random.choice(indicesA, size=n, replace=True)
        y = np.random.choice(indicesB, size=n, replace=True)
        # Get random connection strengths
        c = np.random.normal(loc=mu, scale=sigma, size=n)
        # Change random connections
        self.setConnections(strengths=c, indicesA=x, indicesB=y,
                modulatory=modulatory, setBase=True, setCurrent=True)

    def setConnections(self, strengths, indicesA=None, indicesB=None,
            attributeName=None, attributeNameA=None, attributeValueA=None,
            attributeNameB=None, attributeValueB=None, modulatory=False,
            setBase=True, setCurrent=True):
        '''Set connection strengths in net

        Three options are given to select upstream and downstream neurons to
            set connection strengths between. For upstream neurons, either
            indicesA may be used to directly specify neuron indices, or
            attributeNameA/attributeValueA can be used to filter neurons. For
            downstream neurons, either indicesB or attributeNameB/
            attributeValueB can be used to filter neurons. Additionally,
            attributeName may be used for convenience to specify both upstream
            and downstream attribute name to use for neuron selection.

        Arguments:
            strengths = the connection strengths to set. If this is an iterable,
                there must be one per neuron specified. If this is not an
                iterable, it must be a scalar value that all connections
                will be set to.
            indicesA = (optional) the indices of neurons to choose from to
                randomly connect from. If None, all neurons may be chosen to
                connect from.
            indicesB = (optional) the indices of neurons to choose from to
                randomly connect to. If None, all neurons may be chosen to
                connect to.
            attributeName = (optional) the attribute name to select both
                upstream and downstream neurons.
            attributeNameA = (optional) the attribute used to select the
                upstream neurons. If indicesA or attributeName is supplied, this
                is ignored.
            attributeValueA = (optional) the attribute value to select upstream
                neurons to connect from.
            attributeNameB = (optional) the attribute used to select the
                downstream neurons. If indicesB or attributeName is supplied,
                this is ignored.
            attributeValueA = (optional) the attribute value to select
                downstream neurons to connect from.
            modulatory = (optional) boolean flag indicating that the modulatory
                network instead of the direct network should be randomized.
                Default is false (direct, not modulatory)
            setBase = (optional) boolean flag indicating whether to change the
                base connection strengths. Default is True.
            setCurrent = (optional) boolean flag indicating whether to change
                the current connection strengths. Default is True.
        '''

        if attributeName is not None:
            attributeNameA = attributeName
            attributenameB = attributeName
        if attributeNameB is None and attributeNameA is not None:
            attributeNameB = attributeNameA
        if attributeValueB is None:
            attributeValueB = attributeValueA
        if indicesB is None:
            indicesB = indicesA
        indicesA = self.getIndices(indices=indicesA, attributeName=attributeNameA, attributeValue=attributeValueA)
        indicesB = self.getIndices(indices=indicesB, attributeName=attributeNameB, attributeValue=attributeValueB)

        try:
            if len(indicesA) == 0:
                # Upstream neuron group is empty do nothing.
                return
        except TypeError:
            # Probably a slice object rather than a list of indices, carry on.
            pass
        try:
            if len(indicesB) == 0:
                # Upstream neuron group is empty do nothing.
                return
        except TypeError:
            # Probably a slice object rather than a list of indices, carry on.
            pass

        try:
            # Check if strengths is an array or scalar.
            iter(strengths)
        except TypeError:
            # strengths is a scalar strength value. Create an array out of
            #   this value to match the indices arrays.
            strengths = np.full(indicesA.shape, strengths)

        if modulatory:
            if setBase:
                self.baseModConnections[indicesA, indicesB] = strengths
            if setCurrent:
                self.modConnections[indicesA, indicesB] = strengths
        else:
            if setBase:
                self.baseConnections[indicesA, indicesB] = strengths
            if setCurrent:
                self.connections[indicesA, indicesB] = strengths

    def setThresholds(self, thresholds, indices=None, attributeName=None,
            attributeValue=None, setBase=False, setCurrent=True):
        '''Set the thresholds of the specified neurons.

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Arguments:
            thresholds = a threshold value, or array of threshold values to set
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to set threshold for
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the attribute value to use to select
                neurons to set thresholds for
            setBase = (optional) boolean flag indicating whether to change the
                base threshold. Default is False.
            setCurrent = (optional) boolean flag indicating whether to change
                the current threshold. Default is True.
        '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)
        self.thresholds[indices] = thresholds
        self.baseThresholds[indices] = thresholds

    def setRefractoryPeriods(self, refractoryPeriods, indices=None, attributeName=None, attributeValue=None):
        '''Set the refractory periods of the specified neurons.

        Either indices or attributeName/attributeValue pair can be used to
            select neurons. If no selecting criteria are given, all neurons are
            selected.

        Arguments:
            refractoryPeriods = a refractory period value, or array of
                refractory period values to set
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to set refractory period for
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the attribute value to use to select
                neurons to set refractory period for
        '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)
        self.refractoryPeriods[indices] = refractoryPeriods

    def createChain(self):
        '''An algorithm for adding a chain topology to the net.

            Each neuron will be connected in turn to the neuron with the next
            index, and the last neuron will then be connected to the first.
        '''

        indicesA = np.array(range(self.numNeurons))
        indicesB = np.roll(indicesA, -1)
        self.setConnections(10, indicesA=indicesA, indicesB=indicesB)

    def createLayers(self, nLayers=1, nIntraconnects=1, nInterconnects=1, mu=0, sigma=1, indices=None, attributeName=None, attributeValue=None):
        '''An algorithm for adding a layered topology to part of the net.

        The selected neurons will be assigned to layers. Each layer will be
            self-interconnected, and connected to the next layer. If selection
            criteria are omitted, all neurons will be included in layered
            topology.

        Arguments:
            nLayers = integer number of layers to divide neurons into
            nIntraconnects = number of synapses to randomly form between
                the neurons within each layer
            nInterconnects = number of connections to form between
                adjacent layers. If this is a single integer, then it is the
                number of both forward and backward connections. If it is a
                tuple of integers, the first integer is the number of forward
                connections to make, the second is the number of backwards
                connections to make.
            mu = mean connection strength
            sigma = standard deviation of connection strength
            indices = (optional) either an iterable or slice object indicating
                indices of neurons to include in layered topology
            attributeName = (optional) the attribute name corresponding to the
                attribute values given.
            attributeValue = (optional) the attribute value to use to select
                neurons to include in layered topology
        '''

        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)
        # Get number of selected neurons (index in case we're dealing with a slice)
        numNeurons = len(np.arange(self.numNeurons)[indices])

        if type(nInterconnects) == type(int()):
            # User passed in a single integer. Convert it to
            #   (nForwards, nBackwards) format
            nInterconnects = (nInterconnects, nInterconnects)

        layerNumbers = np.array([np.floor(x/(numNeurons/nLayers)) for x in range(numNeurons)])
        # Give all neurons a layer number of NaN, so they are unselectable by layer number
        self.setAttributes('layer', values=np.nan, indices=np.s_[:])
        # Set layer numbers of specific neurons selected to be in layers to the
        #   correct layer value
        self.setAttributes('layer', values=layerNumbers, indices=indices)
        layerNums = self.getUniqueAttributes('layer')
        for k, layerNum in enumerate(layerNums):
            # Make within-layer connections
            self.randomizeConnections(nIntraconnects, mu, sigma, attributeName='layer', attributeValueA=layerNum)
            if k < len(layerNums)-1:
                # Make interconnections between adjacent layers
                nextLayerNum = layerNums[k+1]
                self.randomizeConnections(nInterconnects[0], mu, sigma, attributeName='layer', attributeValueA=layerNum, attributeValueB=nextLayerNum)
                self.randomizeConnections(nInterconnects[1], mu, sigma, attributeName='layer', attributeValueA=nextLayerNum, attributeValueB=layerNum)

    def arrangeNeurons(self):
        '''Attempt to give neurons spatial positions

        Attempts to cluster neurons according to their connectivity patterns.
        Deprecated.
        '''

        # Try to arrange neurons based on how they're connected? Maybe?
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.connections)
        # Get coordinate scales
        ((minX, maxX), (minY, maxY)) = self.getPositionRanges(principalComponents)
        xRange = maxX - minX
        yRange = maxY - minY
        xRange = 1 if xRange == 0 else xRange
        yRange = 1 if yRange == 0 else yRange
        # Add jitter
        principalComponents[:, 0] = principalComponents[:, 0] + (xRange/20) * (np.random.rand(self.numNeurons)-0.5)
        principalComponents[:, 1] = principalComponents[:, 1] + (yRange/20) * (np.random.rand(self.numNeurons)-0.5)
        # Get new coordinate scales
        ((minX, maxX), (minY, maxY)) = self.getPositionRanges(principalComponents)
        xRange = maxX - minX
        yRange = maxY - minY
        xRange = 1 if xRange == 0 else xRange
        yRange = 1 if yRange == 0 else yRange
        # Scale coordinates to a standard range
        sRange = 100
        principalComponents[:, 1] = (sRange * (principalComponents[:, 1] - minX)/xRange)

        for k in range(self.numNeurons):
            self.neurons[k].setPosition(principalComponents[k, :])

    def getPositionRanges(self, positionData):
        '''Get the x and y range for neuron spatial position values.'''

        # Position data must be a Nx2 array of x,y coordinates
        return ((np.min(positionData[:, 0]), np.max(positionData[:, 0])), (np.min(positionData[:, 1]), np.max(positionData[:, 1])))

    def showNet(self):
        '''Display graphical representation of net.

        Not recommended for large nets - very slow, deprecated.
        '''
        x = [nr.getX() for nr in self.neurons]
        y = [nr.getY() for nr in self.neurons]
        maxC = np.absolute(self.connections).max()
        markerSize = 10
        markerRadius = np.sqrt(markerSize)/20
        ax = plt.axes()
#        ax.set_aspect(1)
        ax.scatter(x, y, s=markerSize)
        ars = patches.ArrowStyle.Simple(tail_width=0.5, head_width=4, head_length=8)
        cs = patches.ConnectionStyle.Arc3(rad=4)
        for k in range(self.numNeurons):
            print('{k} of {n}'.format(k=k, n=self.numNeurons))
            for j in range(self.numNeurons):
                c = self.connections[k][j] / maxC
                if c > 0:
                    color = np.array([0, 1, 0])
                else:
                    color = np.array([1, 0, 0])
                brightness = abs(c)

                p1 = np.array([x[k], y[k]])
                p2 = np.array([x[j], y[j]])

                if k == j:
                    # Autapse
                    distFromZero = np.linalg.norm(p1)
#                    print('Autapse points:', p1, p2)
                    if distFromZero == 0:
                        dirHat = np.array([0, 1])
                    else:
                        dirHat= p1/np.linalg.norm(p1)
                    dir2Hat = dirHat.dot(Net.ROTATION)
                    sideOffset = markerRadius * dir2Hat
                    p1 = p1 + sideOffset
                    p2 = p2 - sideOffset
                    self.connectionArrows[k][j] = patches.FancyArrowPatch(p1, p2, arrowstyle=ars, color=color, alpha=brightness, connectionstyle=cs)
#                    self.connectionArrows[k][j] = plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], color=color, alpha=brightness, head_width=0.1)
                else:
                    # Synapse
                    dir = p2 - p1
#                    print('Synapse points:', p1, p2)
                    distBetweenPoints = np.linalg.norm(dir)
                    if distBetweenPoints == 0:
                        dirHat = np.array([0, 1])
                    else:
                        dirHat= dir/np.linalg.norm(distBetweenPoints)
                    dirHat = dir / np.linalg.norm(dir)
                    dir2Hat = dirHat.dot(Net.ROTATION)
                    sideOffset = markerRadius * dir2Hat
                    forwardOffset = markerRadius * dirHat
                    p1 = p1 + sideOffset + forwardOffset
                    p2 = p2 + sideOffset - forwardOffset
                    self.connectionArrows[k][j] = patches.FancyArrowPatch(p1, p2, arrowstyle=ars, color=color, alpha=brightness)
#                    self.connectionArrows[k][j] = plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], color=color, alpha=brightness, head_width=0.1)
                ax.add_patch(self.connectionArrows[k][j])
        plt.show()

    def run(self, iters, visualize=True):
        '''Simulate net

        Arguments:
            iters = number of simulation iterations to run
            visualize = (optional) visualize simulation (not implemented)
        '''

        if visualize:
            pass
            # plt.ion()
            # plt.show()
        for k in range(iters):
            print('k:', k)
            self.activate()
            if visualize:
                pass

    def runInteraction(self, interactor, inputIndices=None, inputAttributeName=None, inputAttributeValue=None, outputIndices=None, outputAttributeName=None, outputAttributeValue=None):
        '''Run neural network with a pattern stimulation, and return the outputs

        This differs from runSequence - instead of a predefined array of inputs
            for each time point, this takes a Interactor object which can
            deliver an output dependent stimulus. See the Interactor class.

        Arguments:
            interactor = an object that adheres to the same interface as the
                BaseInteractor class. Provides stimulation and target output
            inputIndices = (optional) either an iterable or slice object
                indicating indices of neurons to set activations for
            inputAttributeName = (optional) the attribute name corresponding to
                the attribute values given
            inputAttributeValue = (optional) the attribute value to use to
                select neurons to set activations for
            outputIndices = (optional) either an iterable or slice object
                indicating indices of neurons to get output from
            outputAttributeName = (optional) the attribute name corresponding to
                the attribute values given
            outputAttributeValue = (optional) the attribute value to use to
                select neurons to get output from
        '''

        inputIndices = self.getIndices(indices=inputIndices, attributeName=inputAttributeName, attributeValue=inputAttributeValue)
        outputIndices = self.getIndices(indices=outputIndices, attributeName=outputAttributeName, attributeValue=outputAttributeValue)

        t = 0
        while True:
            try:
                newInput = interactor.next(lastOutput=self.getOutput())
            except StopIteration:
                # Done stimulating
                break;
            print('Interaction #{t}'.format(t=t+1))
            self.addInput(newInput, indices=inputIndices)
            self.activate()
            t = t + 1
            if self.getHistoryLength() < t:
                # Going to lose output if we don't expand history
                self.setHistoryLength(t)

        return self.getOutputSequence(np.s_[:T], indices=outputIndices)

    def runSequence(self, inputs, inputIndices=None, inputAttributeName=None, inputAttributeValue=None, outputIndices=None, outputAttributeName=None, outputAttributeValue=None):
        '''Run neural network with a sequence of inputs, and return the outputs.

        This differs from runPattern - instead of Interactor object, input is
            provided by a predefined array of inputs.

        Arguments:
            inputs = An NxT series of inputs (N=# of input neurons, T=# of time
                steps) to stimulate the net with. This also defines the number
                of time steps to run.
            inputIndices = (optional) either an iterable or slice object
                indicating indices of neurons to set activations for
            inputAttributeName = (optional) the attribute name corresponding to
                the attribute values given
            inputAttributeValue = (optional) the attribute value to use to
                select neurons to set activations for
            outputIndices = (optional) either an iterable or slice object
                indicating indices of neurons to get output from
            outputAttributeName = (optional) the attribute name corresponding to
                the attribute values given
            outputAttributeValue = (optional) the attribute value to use to
                select neurons to get output from
        '''

        inputIndices = self.getIndices(indices=inputIndices, attributeName=inputAttributeName, attributeValue=inputAttributeValue)
        outputIndices = self.getIndices(indices=outputIndices, attributeName=outputAttributeName, attributeValue=outputAttributeValue)

        N, T = inputs.shape
        if not N == len(inputIndices):
            raise IndexError('Error, size of input axis 0 ({S}) should match the number of neurons selected ({N}).'.format(S=N, N=len(inputIndices)))

        if T > self.getHistoryLength():
            print('Expanding net history to accomodate sequence length')
            self.setHistoryLength(T)

        for t in range(T):
            print('{t} of {T}'.format(t=t+1, T=T))
            self.addInput(inputs[:, t], indices=inputIndices)
            self.activate()

        return self.getOutputSequence(np.s_[:T], indices=outputIndices)

class Connectome:
    '''A class representing a set of projecting populations of neurons

    This class represents a particular algorithm for randomly generating neural
    networks (the Net class). A connectome object is meant to be loaded from
    a set of parameters in a CSV file of a particular format (see the
    Connectome.HEADER_ROW attribute for the formatting). It should be fully
    interchangeably and interconvertible with the CSV file

    Each row in the CSV file provides the specification for one "projecting
    population", which is a group of neurons that project to one or more other
    regions. One named "region" can contain one or more projection populations.
    Each projecting population has attributes specifying what numbers, types,
    and strengths of connections to make to the downstream regions, as well
    as neuronal attributes of the population neurons.

    It's possible to load a Connectome object from a CSV file, and to store a
    Connectome object as a CSV file.
    '''

    HEADER_ROW = [  # The header row for the connectome CSV spec file
        'Region name',
        'Downstream region names (comma separated for co-projections)',
        'Population name',
        'Projection proportions (same order as region names)',
        'Number of neurons',
        'Modulatory?',
        'Mean,std neuron threshold',
        'Mean,std refractory time',
        'Mean,std number of connections per neuron',
        'Mean,std connection strength',
    ]

    def __init__(self, connectomeFile=None, connectomeFileObject=None):
        '''Constructor for connectome class

        Either a path to a connectome file, or a connectome file object must
            be provided to read parameters from. For connectome file format
            specification, see example connectome CSV files.

        Arguments:
            connectomeFile = the path to a connectome CSV file.
            connectomeFileObject = a file like object containing connectome
                parameters in CSV format
        '''
        self.populations = []
        # Open the connectome definition file
        if connectomeFile is not None:
            # User provided a file path
            csvFile = open(connectomeFile, newline='')
        elif connectomeFileObject is not None:
            # User provided a file like object
            csvFile = connectomeFileObject
        with csvFile:
            reader = csv.reader(csvFile)
            # Loop over rows and construct population projection objects
            for k, row in enumerate(reader):
                if k == 0:
                    # Header row
                    continue
                self.populations.append(ProjectingPopulation(*row))
        # Get a unique list of region names
        upstreamRegions = set(p.regionName for p in self.populations)
        downstreamRegions = set([region for p in self.populations for region in p.connectedRegions])
        self.regionNames = list(upstreamRegions | downstreamRegions)
        # Create a mapping from region ID ==> region name
        self.regionNameMap = dict(zip(range(len(self.regionNames)), self.regionNames))
        # Create a mapping from region name ==> region ID
        self.regionNameReverseMap = dict(zip(self.regionNames, range(len(self.regionNames))))
        self.populationIDs = []
        self.regionIDs = []

    def createNet(self, **kwargs):
        '''Construct net according to connectome specification

        Arguments:
            Any keyword arguments will be passed through to the Net constructor
        '''

        # Generation neuron count for each population
        numNeurons = []
        totalNumNeurons = 0
        for k in range(len(self.populations)):
            numNeurons.append(int(np.round(np.random.normal(loc=self.populations[k].meanNumNeurons, scale=self.populations[k].stdNumNeurons))))
            totalNumNeurons += numNeurons[-1]

        # Get a list of the population and region IDs of each neuron so we can make them attributes in the net later
        for k in range(len(self.populations)):
            # Create a list of population IDs - each population is within one region, and projects to a set of other regions
            self.populationIDs.extend([k for j in range(numNeurons[k])])
            # Create a list of region IDs - each region is a named group of neurons
            regionID = self.regionNameReverseMap[self.populations[k].regionName]
            self.regionIDs.extend([regionID for j in range(numNeurons[k])])

        n = Net(numNeurons=totalNumNeurons, refractoryPeriodMean=4, refractoryPeriodSigma=3, **kwargs)
        # Set an attribute marking each neuron with its region number and population number
        n.setAttributes('population', values=self.populationIDs)
        n.setAttributes('region',     values=self.regionIDs, attributeMap=self.regionNameMap)

        for k in range(len(self.populations)):
            # Set thresholds
            thresholds = np.random.normal(
                loc=self.populations[k].meanThreshold,
                scale=self.populations[k].stdThreshold,
                size=numNeurons[k])
            n.setThresholds(thresholds, attributeName='population', attributeValue=k, setBase=False, setCurrent=True)
            # Set refractory periods
            refractoryPeriods = np.random.normal(
                loc=self.populations[k].meanRefractoryPeriod,
                scale=self.populations[k].stdRefractoryPeriod,
                size=numNeurons[k])
            n.setRefractoryPeriods(refractoryPeriods, attributeName='population', attributeValue=k)
            # Get a list of region IDs that this population projects to
            connectedRegionIDs = [self.regionNameReverseMap[cr] for cr in self.populations[k].connectedRegions]
            if len(connectedRegionIDs) == 0:
                # This region has no outgoing projections.
                continue
            # Determine how many connections to make
            numConnections = max(0, numNeurons[k] * np.round(np.random.normal(loc=self.populations[k].meanNumConnections, scale=self.populations[k].stdNumConnections)).astype('int'))
            # Choose how many connections will go to each downstream regions
            connections = np.random.choice(connectedRegionIDs, size=numConnections, p=self.populations[k].connectionProbabilities)
            # Loop over each downstream region and add connections
            for j in range(len(connectedRegionIDs)):
                regionalConnectionCount = sum(connections == connectedRegionIDs[j])
                # print('making {n} connections from {a} to {b}'.format(n=regionalConnectionCount, a=k, b=self.regionNameMap[connectedRegionIDs[j]]))
                n.randomizeConnections(regionalConnectionCount,
                    self.populations[k].meanConnectionStrength,
                    self.populations[k].stdConnectionStrength,
                    attributeNameA="population", attributeValueA=k,
                    attributeNameB="region", attributeValueB=connectedRegionIDs[j],
                    modulatory=self.populations[k].modulatory
                    )
        return n

    def encodeConnectomeSpec(self):
        '''Convert the connectome back into connectome CSV text specification

        Returns:
            list of rows, where each row is a list of strings representing cells
        '''

        rows = []
        for population in self.populations:
            rows.append(population.encodePopulationSpec())
        return rows

    def copy(self):
        '''Return an independent copy of this Connectome object'''

        connectomeCopy = None
        with io.StringIO() as f:
            self.streamToFile(f)
            connectomeCopy = Connectome(f)

        return connectomeCopy

    def streamToFile(self, fileObject):
        '''Write connectome data to a text file object in CSV format

        See the sample connectome CSV files for the expected file format.

        Arguments:
            fileObject = a file-like object to save to.
        '''

        rows = self.encodeConnectomeSpec()
        writer = csv.writer(fileObject)
        writer.writerow(Connectome.HEADER_ROW)
        for row in rows:
            writer.writerow(row)

    def save(self, file):
        '''Save connectome as a connectome spec CSV file

        See the sample connectome CSV files for the expected file format.

        Arguments:
            file = string or Path representing the path to a CSV file to save
                connectome spec to
        '''

        with open(file, 'w', newline='') as f:
            self.streamToFile(f)

    def mutate(self, noProjectRegions=[], immutablePopulationIndices=[]):
        '''Randomly make changes in the population parameters

        Arguments:
            noProjectRegions = (optional) a list of region names which may not
                be projected to
            immutablePopulationIndices = (optional) a list of population indices
                that are immutable - may not be mutated, or removed
        '''

        # Choose a population index to mutate
        mutablePopulations = [idx for idx in range(len(self.populations)) if idx not in immutablePopulationIndices]
        if len(mutablePopulations) == 0:
            # Could change this so it generates a new randomized population
            print('No mutable populations')
            return

        # Pick a population to mutate
        popIndex = np.random.choice(mutablePopulations)
        pop = self.populations[popIndex]

        # Define mutable parameters - how should we mutate this population?
        mutableParameters = [
            'connectedRegions',
            'connectionProbabilities',
            'meanThreshold',
            'stdThreshold',
            'meanRefractoryPeriod',
            'stdRefractoryPeriod',
            'meanNumConnections',
            'stdNumConnections',
            'meanConnectionStrength',
            'stdConnectionStrength',
            'duplicatePopulation',
            'modulatory',
            'removePopulation'
        ]

        weights = np.array([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1
        ])

        probabilities = weights / np.sum(weights)

        if len(pop.connectedRegions) == 0:
            # Can't mutate connection probabilities if there are no connected regions
            mutableParameters.remove('connectionProbabilities')

        # Choose how to mutate the chosen population
        param = np.random.choice(mutableParameters, p=probabilities)

        if param == "connectedRegions":
            # Mutate regions that are projected to (form a new projection to a
            #   new region, or prune a projection)
            allowableRegions = [r for r in self.regionNames if r not in noProjectRegions]
            numConnectedRegions = len(pop.connectedRegions)
            if numConnectedRegions == len(allowableRegions):
                # Max # of connected regions. Remove one
                regionDelta = -1
            elif numConnectedRegions == 0:
                # Min # of connected regions. Add one
                regionDelta = 1
            else:
                # Calculate probability of add vs remove
                #   Add is more likely if # of connected regions is low
                #   Remove is more likely is # of connected regions is high
                #   If # of connected regions is equal to half the number of
                #   allowable connected regions, probability is 50/50
                deltaProbs = np.array([numConnectedRegions, len(allowableRegions) - numConnectedRegions])
                deltaProbs = deltaProbs / np.sum(deltaProbs)
                # Choose how many to add/remove
                regionDelta = np.random.choice([-1, 1], p=deltaProbs)
            if regionDelta > 0:
                # Add one or more regions
                unusedRegions = [r for r in allowableRegions if r not in pop.connectedRegions]
                newConnectedRegions = np.random.choice(unusedRegions, size=regionDelta)
                pop.connectedRegions.extend(newConnectedRegions)
                # Generate new connection probabilties
                newConnectionProbabilties = np.random.random(size=regionDelta)
                pop.connectionProbabilities = np.append(pop.connectionProbabilities * numConnectedRegions, newConnectionProbabilties)
                pop.connectionProbabilities = pop.connectionProbabilities / np.sum(pop.connectionProbabilities)
            else:
                # Remove one or more regions
                removableRegionIndices = [idx for idx in range(numConnectedRegions) if idx not in immutablePopulationIndices]
                indicesToRemove = np.random.choice(removableRegionIndices, size=-regionDelta)
                pop.connectedRegions = [r for j, r in enumerate(pop.connectedRegions) if j not in indicesToRemove]
                pop.connectionProbabilities = np.delete(pop.connectionProbabilities, indicesToRemove)
        elif param == "connectionProbabilities":
            # Pick which probability to change
            j = np.random.choice(range(len(pop.connectionProbabilities)))
            prob = pop.connectionProbabilities[j]
            # Provide lower limit of 0.1 to scale of random mutation distribution
            scale = max(0.1, prob/2)
            # Compute new randomly changed probability
            newProb = np.random.normal(loc=prob, scale=scale)
            pop.connectionProbabilities[j] = newProb
            # Renormalize
            pop.connectionProbabilities /= sum(pop.connectionProbabilities)
        elif param in ["meanThreshold", "stdThreshold", "meanRefractoryPeriod",
            "stdRefractoryPeriod", "meanNumConnections", "stdNumConnections",
            "meanConnectionStrength", "stdConnectionStrength"]:
            # Mutate one of the purely numerical parameters
            val = getattr(pop, param)
            # Compute scale of random variation - half the value, but at least 1.
            scale = max(1, abs(val)/2)
            newVal = np.random.normal(loc=val, scale=scale)
            setattr(pop, param, newVal)
        elif param == 'modulatory':
            pop.modulatory = not pop.modulatory
        elif param in ['duplicatePopulation']:
            # Duplicate the population
            newPop = pop.copy()
            suffixMatch = re.search('.*?([0-9]+)', newPop.populationName)
            if suffixMatch:
                suffix = suffixMatch.group(1)
                basePopulationName = newPop.populationName[:-len(suffix)]
            else:
                suffix = '0'
                basePopulationName = newPop.populationName
            newPop.populationName = basePopulationName + str(int(suffix)+1)
            self.populations.append(newPop)
        elif param in ['removePopulation']:
            self.populations.pop(popIndex)

def boolParser(value):
    '''Parse a value as a boolean, allowing for strings "0" and "1"'''
    if type(value) == type(str()):
        return bool(int(value))
    else:
        return bool(value)

class ProjectingPopulation:
    '''A class representing one projecting population within a connectome spec'''
    def __init__(self, regionA, regionsB, populationName, proportions, numNeurons, modulatory, thresholds, refractoryPeriods, numConnections, connectionStrength):
        '''Constructor for ProjectingPopulation class

        This is meant to be directly passed the raw string elements in a single
            row of a connectome specification CSV file.

        Arguments:
            regionA = the name of the upstream region that contains this
                projecting population
            regionsB = a comma-separated string of names of downstream regions
                that this projecting population projects to
            populationName = a name for this projecting population
            proportions = a comma-separated string containing relative weights
                for the downstream regions. Must have the same # of proportions
                as the # of regions in regionsB. The connections will be
                distributed according to these weights.
            numNeurons = a string containing a comma-separated pair of numbers
                indicating the mean and stdev of the number of neurons in this
                population, or a tuple of numerical values representing the mean
                and stdev
            modulatory = a boolean flag indicating whether the connections are
                modulatory. This can be expressed as a string "0" or "1", or
                as any other boolean-castable python type.
            thresholds = a string containing a comma-separated pair of numbers
                indicating the mean and stdev of the thresholds for the
                neurons in this population, or a tuple of numerical values
                representing the mean and stdev
            refractoryPeriods = a string containing a comma-separated pair of
                numbers indicating the mean and stdev of the refractory periods
                for the neurons in this population, or a tuple of numerical
                values representing the mean and stdev.
            numConnections = a string containing a comma-separated pair of
                numbers indicating the mean and stdev of the number of
                outgoing connections per neuron, or a tuple of numerical values
                representing the mean and stdev.
            connectionStrength = a string containing a comma-separated pair of
                numbers indicating the mean and stdev of the connection
                strengths for the outgoing connections, or a tuple of numerical
                values representing the mean and stdev.
        '''

        # Parse all text fields into usable population attributes
        self.regionName = regionA
        self.populationName = populationName
        (self.meanNumNeurons,self.stdNumNeurons) = self.unpackParam(numNeurons, parser=int)
        self.connectedRegions = self.unpackParam(regionsB, parser=str)
        if len(self.connectedRegions) == 0:
            # No downstream connected regions given
            self.meanNumConnections = 0
            self.stdNumConnections = 0
            self.meanConnectionStrength = 0
            self.stdConnectionStrength = 0
        else:
            (self.meanNumConnections, self.stdNumConnections) = self.unpackParam(numConnections, parser=float)
            (self.meanConnectionStrength, self.stdConnectionStrength) = self.unpackParam(connectionStrength, parser=float)
        proportions = np.array(self.unpackParam(proportions, parser=float))
        if len(proportions) == 0:
            # No proportions given
            self.connectionProbabilities = np.array([1])
        else:
            self.connectionProbabilities = proportions / np.sum(proportions)
        (self.modulatory,) = self.unpackParam(modulatory, parser=boolParser)
        (self.meanThreshold, self.stdThreshold) = self.unpackParam(thresholds, parser=float)
        (self.meanRefractoryPeriod, self.stdRefractoryPeriod) = self.unpackParam(refractoryPeriods, parser=float)

    def __str__(self):
        connections = '|'.join('{r}({p:.02f})'.format(r=r, p=p) for r, p in zip(self.connectedRegions, self.connectionProbabilities))
        return 'N={mn:.02f}+/-{sn:.02f} {r}==>{c} N={mc:.02f}+/-{sc:.02f} strength={ms:.02f}+/-{ss:.02f} refractory={mr:.02f}+/-{sr:.02f} threshold={mt:.02f}+/-{st:.02f}'.format(
            mn=self.meanNumNeurons, sn=self.stdNumNeurons,
            r=self.regionName, c=connections, mc=self.meanNumConnections,
        )

    def copy(self):
        return ProjectingPopulation(*self.encodePopulationSpec())

    def encodePopulationSpec(self):
        '''Convert the parameters back into a population specification'''

        # Convert all attributes into text fields with the appropriate
        #   formatting and grouping
        regionA = self.regionName
        regionsB = ','.join(self.connectedRegions)
        populationName = self.populationName
        proportions = ','.join([str(p) for p in self.connectionProbabilities])
        numNeurons = '{mu},{sigma}'.format(mu=self.meanNumNeurons, sigma=self.stdNumNeurons)
        modulatory = '1' if self.modulatory else '0'
        thresholds = '{mu},{sigma}'.format(mu=self.meanThreshold, sigma=self.stdThreshold)
        refractoryPeriods = '{mu},{sigma}'.format(mu=self.meanRefractoryPeriod, sigma=self.stdRefractoryPeriod)
        numConnections = '{mu},{sigma}'.format(mu=self.meanNumConnections, sigma=self.stdNumConnections)
        connectionStrength = '{mu},{sigma}'.format(mu=self.meanConnectionStrength, sigma=self.stdConnectionStrength)

        # Return all text fields in the expected order
        return regionA, regionsB, populationName, proportions, numNeurons, modulatory, thresholds, refractoryPeriods, numConnections, connectionStrength

    def unpackParam(self, params, parser=float):
        '''Unpack parameters

        param = a parameter, either as a comma-separated string, or a tuple
            of values
        parser = a function that parses a parameter value. Ignored if params are
            a tuple, instead of a comma-separated string'''
        if type(params) == type(str()):
            splitParams = [p.strip() for p in params.split(',')]
            if len(splitParams) == 1 and len(splitParams[0]) == 0:
                # params is empty
                return []
            else:
                return [parser(s) for s in splitParams]
        elif type(params) == type(tuple()):
            return params

class NetViewer:
    '''Class allowing for visualization of nets'''
    def __init__(self, root):
        self.root = root

class Neuron:
    ''' A class for holding auxiliary attributes for a single neuron in a net.'''
    def __init__(self, net, index, x=None, y=None):
        '''Constructor for the Neuron class

        Arguments:
            net = the Net that this Neuron belongs to
            index = the index within the net of this neuron
            x = (optional) the spatial x position of the neuron
            y = (optional) the spatial y position of the neuron
        '''

        self.net = net
        self.index = index
        self.position = [x, y]

    def setX(self, x):
        '''Set the spatial x position of the neuron

        x = the spatial x position of the neuron
        '''

        self.position[0] = x

    def setY(self, y):
        '''Set the spatial y position of the neuron

        y = the spatial y position of the neuron
        '''

        self.position[1] = y
    def setPosition(self, position):
        '''Set the spatial position of the neuron

        Arguments:
            position = a tuple of numbers representing the the spatial x and y
                positions of the neuron
        '''

        self.position = position
    def getX(self):
        '''Get the spatial x position of the neuron

        Returns:
            x = the spatial x position of the neuron
        '''
        return self.position[0]
    def getY(self):
        '''Get the spatial y position of the neuron

        Returns:
            y = the spatial y position of the neuron
        '''
        return self.position[1]

    def setActivation(self, activation):
        '''Convenience function to set the activation of this neuron in the net

        Arguments:
            activation = the activation level to set the neuron's activation to
        '''

        self.net.activations[self.index] = activation

    def getActivation(self):
        '''Convenience function to get the activation of this neuron in the net

        Returns:
            activation = the activation level of this neuron
        '''

        self.net.activations[self.index] = activation

if __name__ == "__main__":
    '''Code to run when this module is run directly, rather than imported

    Preset run methods:

    1. Create example layer topology:

        python net.py layers

    2. Use a connectome file:

        python net.py connectomeFileName.csv

    3. Create example chain topology:

        python net.py chain

    '''

    np.set_printoptions(linewidth=100000, formatter=dict(float=lambda x: "% 0.1f" % x))

    initType = sys.argv[1]   #'connectome'

    # Create blank stimulation matrix (neuron x time)
    nInputs = 50
    T = 500
    stim = np.zeros([nInputs, T])
    inIdx = range(nInputs)

    # Display summed activity over this grouping type:
    majorGrouping = None
    if initType == 'layers':
        N = 1000
        NL = N//100
        n = Net(N, refractoryPeriodMean=10, refractoryPeriodSigma=7, historyLength=T)
        regularIndices = np.arange(0, 1000)
        modulatoryIndices = np.arange(800, N)
        n.createLayers(nLayers=NL,  nIntraconnects=N, nInterconnects=N//10, mu=0.7, sigma=2, indices=regularIndices)
        stim[0:10, 0] = 10
        majorGrouping = 'layer'
    elif initType == "connectome":
        connectomeFile = sys.argv[2]
        co = Connectome(connectomeFile)
        n = co.createNet(historyLength=T)
        stim[0:10, 0] = 10
        majorGrouping = 'region'
    elif initType == "stim":
        connectomeFile = sys.argv[2]
        print('Running stimulation demo with connectome {c}'.format(c=connectomeFile))
        co = Connectome(connectomeFile)
        n = co.createNet(historyLength=T)
        stim[:, ::10] = 10
        inIdx = range(nInputs)
        majorGrouping = 'region'
    elif initType == "chain":
        N = 300
        n = Net(N, refractoryPeriodMean=10, refractoryPeriodSigma=7, historyLength=T)
        n.createChain()
        stim[0, 0] = 10

    nOutputs = 26
    outIdx = range(n.numNeurons)[-nOutputs:]

    output = n.runSequence(stim, inputIndices=inIdx, outputIndices=outIdx)

    f, axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1]})
    axs[0, 0].imshow(np.flip(n.history, axis=1), cmap='binary', interpolation='none') #, 'XData', np.arange(n.historyLength))
    axs[0, 0].set_aspect('auto')
    # for k in range(n.numNeurons):
    #     plt.step(np.arange(n.historyLength), 5*n.history[k, :] + 10*k)
    # for k in range(n.numNeurons):
    #     for j in range(n.numNeurons):
    #         print('{c:0.01f} '.format(c=n.connections[k][j]), end='')
    #     print()
    axs[0, 0].set_ylabel('Neuron #')
    axs[0, 0].set_xlabel('Simulated time')
    axs[0, 0].title.set_text('Raster')
    connIm = axs[0, 1].imshow(n.connections, cmap='seismic', interpolation='none')
    axs[0, 1].set_aspect('auto')
    cRadius = max(np.max(n.connections), abs(np.min(n.connections)))
    connIm.set_clim(-cRadius, cRadius)
    axs[0, 1].set_ylabel("Upstream neuron #")
    axs[0, 1].set_xlabel("Downstream neuron #")
    axs[0, 1].title.set_text('Connection matrix')

    if majorGrouping is not None:
        groupNums = n.getUniqueAttributes(majorGrouping)
        for groupNum in groupNums:
            groupIdx = n.filterByAttribute(majorGrouping, groupNum)
            if len(groupIdx) == 0:
                continue
            axs[1, 0].plot(10*np.mean(np.flip(n.history[groupIdx, :], axis=1), axis=0) + (max(groupNums) - groupNum)*2)
            axs[1, 0].title.set_text('Region-summed firing rate')

    axs[1, 1].imshow(stim, cmap='binary', interpolation='none')
    axs[1, 1].set_aspect('auto')
    axs[1, 1].title.set_text('Stimulation pattern')

    plt.show()
