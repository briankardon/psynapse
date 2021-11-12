import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import csv
import sys

class Net:
    ROTATION = np.array([[0, -1], [1, 0]])
    def __init__(self, numNeurons=1, refractoryPeriods=None, refractoryPeriodMean=4, refractoryPeriodSigma=3, thresholds=None, thresholdMean=1, thresholdSigma=0, historyLength=700):
        # A class representing a simulated neural network. Neurons are
        #   are initially unconnected. Apply a connection algorithm to
        #   connect the neurons.
        #   numNeurons = # of neurons to create within the net
        #   refractoryPeriodMean = the mean random refractory period to give the
        #       neurons.
        #   refractoryPeriodMean = the standard deviation of the random
        #       refractory period to give the neurons
        #   historyLength = the amount of simulated time to save recorded firing
        #       patterns

        self.historyLength = historyLength
        self.numNeurons = numNeurons
        self.connections = np.zeros([self.numNeurons, self.numNeurons])   # Direct excitatory/inhibitory connection matrix
        self.modConnections = np.zeros([self.numNeurons, self.numNeurons])  # Modulatory connection matrix
        self.neurons = [Neuron(self, k) for k in range(self.numNeurons)]
        self.activations = np.zeros(self.numNeurons)
        self.history = np.zeros([self.numNeurons, self.historyLength])
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
        pass

    def loadNet(self, filename):
        pass

    def addAttribute(self, name, indices=None, initialValues=None, attributeMap=None):
        # name = string, name of new attribute
        # initialValues = list of initial values, or None to initialize with all
        #   zeros. If a list, it must have length equal to numNeurons
        # attributeMap = a dictionary mapping attribute values to some kind of human readable name
        #   or None, indicating there is no mapping.
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
        # Deletes the specified attribute.
        try:
            idx = self.attributeNames.index(name)
        except ValueError():
            raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        self.attributeNames.pop(idx)
        self.attributeMaps.pop(idx)
        self.attributeMapsReversed.pop(idx)
        np.delete(self.attributeValues, idx, 0)

    def getAttributes(self, name, indices=None, mapped=False):
        # Return a value or array of values corresponding to the attribute
        #   specified by name and the neuron indices specified by indices.
        #   If indices is None, this returns attribute values for all neurons.
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

    def filterByAttribute(self, name, value):
        # Get a list of neuron indices that match the value. If value is
        #   numerical, return indices of neurons whose attribute value equals
        #   that value exactly. If value is a function, it should take a numpy
        #   array of values and return a numpy array of logicals indicating
        #   which neurons should be selected
        values = self.getAttributes(name)
        try:
            idx = value(values)
        except TypeError:
            idx = np.arange(self.numNeurons)[np.equal(values, value)]
        return idx

    def getUniqueAttributes(self, name):
        return np.unique(self.getAttributes(name))

    def setAttributes(self, name, values=0, valueNames=None, indices=None, addIfNonexistent=True, attributeMap=None):
        # Set a value or array of values corresponding to the attribute
        #   specified by name and the neuron indices specified by indices. The
        #   array of values must be the same size as the array of indices.
        #   If indices is None, this sets attribute values for all neurons.
        # name = the name of an attribute
        # values = a list of values to set. Ignored if valueNames it not None
        # valueNames = a list of value names to set - will be mapped to values first if a attributeMap is available
        # indices = a list of indexes of neurons to set attribute values for
        # addIfNonexistent = a boolean indicating whether or not to create the attribute if it doesn't exist
        # attributeMap = a map between attribute values and value names, only used if we're creating a new attribute
        try:
            idx = self.attributeNames.index(name)
        except ValueError:
            if addIfNonexistent:
                idx = self.addAttribute(name, attributeMap=attributeMap)
            else:
                raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
        if valueNames is not None:
            # Map value names to values, then set them
            values = [self.attributeMapsReversed[valueName] for valueName in valueNames]
        self.attributeValues[idx, indices] = values

    def activate(self):
        # Simulate activation of neurons and transmission of action potentials
        self.history = np.roll(self.history, 1, axis=1)
        # Determine which neurons will fire
        firing = (self.refractoryCountdowns <= 0) & (self.activations > self.thresholds)
        notFiring =  np.logical_not(firing)
        # Pass signal from firing neurons downstream
        self.activations = firing.dot(self.connections)
        # Reset fired neurons to max countdown, continuing decrementing the rest.
        self.refractoryCountdowns = (self.refractoryPeriods * firing) + ((self.refractoryCountdowns - 1) * notFiring)
        # Pass modulatory signals. Add the modulatory signal to the downstream
        #   neuron's threshold, then move all threshold towards each neuron's
        #   base threshold.
        self.thresholds = self.thresholds + firing.dot(self.modConnections) + 0.25*(self.baseThresholds - self.thresholds)
        # Add firing to history
        self.history[:, 0] = firing

    def getIndices(self, indices=None, attributeName=None, attributeValue=None):
        # Get a list of selected neuron indices. If indices is supplied, it is
        #   just directly returned. If attributeName/Value is supplied, indices
        #   of the neurons corresponding to that name/value are returned.
        #   If no arguments are supplied, a slice object selecting all indices
        #   is returned.
        if attributeName is not None:
            indices = self.filterByAttribute(attributeName, attributeValue)
        if indices is None:
            indices = np.arange(self.numNeurons)
        return indices

    def getOutput(self, indices=None, attributeName=None, attributeValue=None):
        # Get activations of neurons indicated by indices.
        #   indices must be either an iterable or slice object indicating which
        #       neurons to get output from, or None, indicating all activations
        #       should be taken.
        #   attributeName = the name of an attribute to use to select neurons
        #   attributeValue = the value of the chosen attribute with which to
        #       select neurons to return output from
        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        return self.activations[indices]

    def stimByAttribute(self, attributeName, attributeValues):
        idx = self.filterByAttribute(attributeName)
        self.addInput(attributeValues, indices=idx)

    def recordByAttribute(self, attributeName, attributeValue):
        idx = self.filterByAttribute(attributeName, attributeValue)
        return self.getOutput(indices=idx)

    def addInput(self, inputs, indices=None, attributeName=None, attributeValue=None):
        # Set activations of neurons indicated by indices to the values in inputs
        #   inputs must be an iterables of less than or equal length to
        #       numNeurons indicating how much to add to the neuron's activation
        #       input
        #   indices must be either an iterable of the same length as inputs
        #       indicating which neurons to activate, or None, indicating the
        #       inputs should just be applied to the first N neurons
        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)

        self.activations[indices] = inputs

    def randomizeConnections(self, n, mu, sigma, indicesA=None, indicesB=None, attributeName=None, attributeNameA=None, attributeValueA=None, attributeNameB=None, attributeValueB=None, modulatory=False):
        # Change random connections
        #   n = number of connections to change
        #   mu = mean connection strength
        #   sigma = standard deviation of connection strength
        #   indicesA = the indices of neurons to choose from to randomly connect
        #       from. If None, all neurons may be chosen to connect from.
        #   indicesB = the indices of neurons to choose from to randomly connect
        #       to. If None, all neurons may be chosen to connect to.
        #   attributeName = an alternate way to select neurons, by attribute
        #       name. If supplied, this name is used to select both upstream
        #       and downstream neurons.
        #   attributeNameB = the attribute used to select the upstream
        #       neurons, if different from the attribute used to select the
        #       downstream neurons. If attributeName is supplied, it is used for
        #       selecting both upstream and downstream neurons.
        #   attributeValueA = the value of the indicated attribute to select
        #       neurons to connect from.
        #   attributeNameB = the attribute used to select the downstream
        #       neurons, if different from the attribute used to select the
        #       upstream neurons. If attributeName is supplied, it is used for
        #       selecting both upstream and downstream neurons.
        #   attributeValueB = the value of the indicated attribute to select
        #       neurons to connect to. If None, the same group of neurons
        #       are selected for connections from and to.
        #   modulatory = boolean flag indicating that the modulatory network
        #       instead of the direct network should be randomized.

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
        if modulatory:
            self.modConnections[x, y] = c
        else:
            self.connections[x, y] = c

    def setThresholds(self, thresholds, indices=None, attributeName=None, attributeValue=None):
        # Set the thresholds of the specified neurons.
        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)
        self.thresholds[indices] = thresholds

    def setRefractoryPeriods(self, refractoryPeriods, indices=None, attributeName=None, attributeValue=None):
        # Set the thresholds of the specified neurons.
        indices = self.getIndices(indices=indices, attributeName=attributeName, attributeValue=attributeValue)
        self.refractoryPeriods[indices] = refractoryPeriods

    def createChain(self):
        for k in range(self.numNeurons-1):
            n.connections[k, k+1] = 10
        n.connections[-1, 0] = 10

    def createLayers(self, nLayers=1, nConnectionsPerLayer=1, nInterconnects=1, mu=0, sigma=1, indices=None, attributeName=None, attributeValue=None):
        # Add a layered topology to the net.
        #   nLayers = number of layers to divide neurons into
        #   nConnectionsPerLayer = number of synapses to randomly form between
        #       the neurons in each layer
        #   nInterconnects = number of connections to form between adjacent
        #       layers. If this is a single integer, then it is the number of
        #       both forward and backward connections. If it is a tuple of
        #       integers, the first integer is the number of forward
        #       connections to make, the second is the number of backwards
        #       connections to make.
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
            self.randomizeConnections(nConnectionsPerLayer, mu, sigma, attributeName='layer', attributeValueA=layerNum)
            if k < len(layerNums)-1:
                # Make interconnections between adjacent layers
                nextLayerNum = layerNums[k+1]
                self.randomizeConnections(nInterconnects[0], mu, sigma, attributeName='layer', attributeValueA=layerNum, attributeValueB=nextLayerNum)
                self.randomizeConnections(nInterconnects[1], mu, sigma, attributeName='layer', attributeValueA=nextLayerNum, attributeValueB=layerNum)

    def downRegulateAutapses(self, factor):
        # Reduce the strength of autapses
        #   factor = the factor by which the mean autapse strength should be
        #       smaller than the mean synapse strength
        x = range(self.numNeurons)

    def getPositionRanges(self, positionData):
        # Position data must be a Nx2 array of x,y coordinates
        return ((np.min(positionData[:, 0]), np.max(positionData[:, 0])), (np.min(positionData[:, 1]), np.max(positionData[:, 1])))

    def arrangeNeurons(self):
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

    def showNet(self):
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
        if visualize:
            pass
            # plt.ion()
            # plt.show()
        for k in range(iters):
            print('k:', k)
            self.activate()
            if visualize:
                pass

class Connectome:
    # A class representing a set of projecting populations
    def __init__(self, connectomeFile):
        self.populations = []
        # Open the connectome definition file
        with open(connectomeFile, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Loop over rows and construct population projection objects
            for k, row in enumerate(reader):
                if k == 0:
                    # Header row
                    continue
                self.populations.append(ProjectingPopulation(*row))
        # Calculate the total number of neurons
        print([p.numNeurons for p in self.populations])
        self.numNeurons = sum([p.numNeurons for p in self.populations])
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
        # Get a list of the population and region IDs of each neuron so we can make them attributes in the net later
        for k in range(len(self.populations)):
            # Create a list of population IDs - each population is within one region, and projects to a set of other regions
            self.populationIDs.extend([k for j in range(self.populations[k].numNeurons)])
            # Create a list of region IDs - each region is a named group of neurons
            regionID = self.regionNameReverseMap[self.populations[k].regionName]
            self.regionIDs.extend([regionID for j in range(self.populations[k].numNeurons)])
        print('populationIDs')
        print(self.populationIDs)
        print('regionIDs')
        print([self.regionNameMap[id] for id in self.regionIDs])

    def createNet(self, **kwargs):
        n = Net(numNeurons=self.numNeurons, refractoryPeriodMean=4, refractoryPeriodSigma=3, **kwargs)
        # Set an attribute marking each neuron with its region number and population number
        n.setAttributes('population', values=self.populationIDs)
        n.setAttributes('region',     values=self.regionIDs, attributeMap=self.regionNameMap)

        for k in range(len(self.populations)):
            # Set thresholds
            thresholds = np.random.normal(
                loc=self.populations[k].meanThreshold,
                scale=self.populations[k].stdThreshold,
                size=self.populations[k].numNeurons)
            n.setThresholds(thresholds, attributeName='population', attributeValue=k)
            # Set refractory periods
            refractoryPeriods = np.random.normal(
                loc=self.populations[k].meanRefractoryPeriod,
                scale=self.populations[k].stdRefractoryPeriod,
                size=self.populations[k].numNeurons)
            n.setRefractoryPeriods(refractoryPeriods, attributeName='population', attributeValue=k)
            # Get a list of region IDs that this population projects to
            connectedRegionIDs = [self.regionNameReverseMap[cr] for cr in self.populations[k].connectedRegions]
            if len(connectedRegionIDs) == 0:
                # This region has no outgoing projections.
                continue
            # Determine how many connections to make
            numConnections = self.populations[k].numNeurons * np.round(np.random.normal(loc=self.populations[k].meanNumConnections, scale=self.populations[k].stdNumConnections)).astype('int')
            # Choose how many connections will go to each downstream regions
            connections = np.random.choice(connectedRegionIDs, size=numConnections, p=self.populations[k].connectionProbabilities)
            # Loop over each downstream region and add connections
            for j in range(len(connectedRegionIDs)):
                regionalConnectionCount = sum(connections == connectedRegionIDs[j])
                print('making {n} connections from {a} to {b}'.format(n=regionalConnectionCount, a=k, b=self.regionNameMap[connectedRegionIDs[j]]))
                n.randomizeConnections(regionalConnectionCount,
                    self.populations[k].meanConnectionStrength,
                    self.populations[k].stdConnectionStrength,
                    attributeNameA="population", attributeValueA=k,
                    attributeNameB="region", attributeValueB=connectedRegionIDs[j]
                    )
        return n

class ProjectingPopulation:
    def __init__(self, regionA, regionsB, populationName, proportions, numNeurons, modulatory, thresholds, refractoryPeriods, numConnections, connectionStrength):
        # print('regionA=', regionA)
        # print('regionsB=', regionsB)
        # print('populationName=', populationName)
        # print('proportions=', proportions)
        # print('numNeurons=', numNeurons)
        # print('modulatory=', modulatory)
        # print('thresholds=', thresholds)
        # print('refractoryPeriods=', refractoryPeriods)
        # print('numConnections=', numConnections)
        # print('connectionStrength=', connectionStrength)
        self.regionName = regionA
        self.populationName = populationName
        self.numNeurons = int(numNeurons)
        if len(regionsB.strip()) == 0:
            # No downstream connected regions given
            self.connectedRegions = []
            self.meanNumConnections = 0
            self.stdNumConnections = 0
            self.meanConnectionStrength = 0
            self.stdConnectionStrength = 0
        else:
            self.connectedRegions = [r.strip() for r in regionsB.split(',')]
            self.meanNumConnections, self.stdNumConnections = [float(n) for n in numConnections.split(',')]
            self.meanConnectionStrength, self.stdConnectionStrength = [float(s) for s in connectionStrength.split(',')]
        if len(proportions.strip()) == 0:
            # No proportions given
            self.connectionProbabilities = [1]
        else:
            proportions = [float(p) for p in proportions.split(',')]
            self.connectionProbabilities = [p/sum(proportions) for p in proportions]
        self.modulatory = bool(modulatory)
        self.meanThreshold, self.stdThreshold = [float(t) for t in thresholds.split(',')]
        self.meanRefractoryPeriod, self.stdRefractoryPeriod = [float(r) for r in refractoryPeriods.split(',')]

class NetViewer:
    def __init__(self, root):
        self.root = root

class Neuron:
    def __init__(self, net, index, x=None, y=None):
        # net is the Net that this Neuron belongs to
        # index is the index within the net that
        self.net = net
        self.index = index
        self.position = [x, y]

    def setX(self, x):
        self.position[0] = x
    def setY(self, y):
        self.position[1] = y
    def setPosition(self, position):
        self.position = position
    def getX(self):
        return self.position[0]
    def getY(self):
        return self.position[1]

    def setActivation(self, activation):
        self.net.activations[self.index] = activation

    def getActivation(self):
        self.net.activations[self.index] = activation

if __name__ == "__main__":
    np.set_printoptions(linewidth=100000, formatter=dict(float=lambda x: "% 0.1f" % x))

    initType = sys.argv[1]   #'connectome'

    # Display summed activity over this grouping type:
    majorGrouping = None
    if initType == 'layers':
        N = 1000
        NL = N//100
        n = Net(N, refractoryPeriodMean=10, refractoryPeriodSigma=7)
        regularIndices = np.arange(0, 1000)
        modulatoryIndices = np.arange(800, N)
        n.createLayers(nLayers=NL,  nConnectionsPerLayer=N, nInterconnects=N//10, mu=0.7, sigma=2, indices=regularIndices)
        majorGrouping = 'layer'
    elif initType == "connectome":
        connectomeFile = 'TestConnectome.csv'
        pcg = Connectome(connectomeFile)
        n = pcg.createNet()
        majorGrouping = 'region'
    elif initType == "chain":
        N = 300
        n = Net(N, refractoryPeriodMean=10, refractoryPeriodSigma=7)
        n.createChain()
#        n.randomizeConnections(N//10, 0, 20)


    # n.randomizeConnections(N*10, 0, 20, indicesA=modulatoryIndices, indicesB=regularIndices, modulatory=True)
    # n.randomizeConnections(N*10, 0, 20, indicesA=regularIndices, indicesB=modulatoryIndices, modulatory=True)
    #     n.randomizeConnections(N, 0.5, 2, attribute='layer', attributeValueA=0, attributeValueB=layerNum)

    # layerNums = n.getUniqueAttributes('layer')
    # for layerNum in layerNums:

#    n.randomizeConnections(100, 2, 1)

    # n.randomizeConnections(N*N, 0, 2)
#    n.arrangeNeurons()
#    n.showNet()
    stim = np.zeros(n.numNeurons)
    stim[0:10] = 10
    n.addInput(stim)
    n.run(n.historyLength)

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
    connIm = axs[0, 1].imshow(n.connections, cmap='seismic', interpolation='none')
    axs[0, 1].set_aspect('auto')
    cRadius = max(np.max(n.connections), abs(np.min(n.connections)))
    connIm.set_clim(-cRadius, cRadius)
    axs[0, 1].set_ylabel("Upstream neuron #")
    axs[0, 1].set_xlabel("Downstream neuron #")

    if majorGrouping is not None:
        groupNums = n.getUniqueAttributes(majorGrouping)
        for groupNum in groupNums:
            groupIdx = n.filterByAttribute(majorGrouping, groupNum)
            if len(groupIdx) == 0:
                continue
            axs[1, 0].plot(10*np.mean(np.flip(n.history[groupIdx, :], axis=1), axis=0) + (max(groupNums) - groupNum)*2)

    plt.show()
