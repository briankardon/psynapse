import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class Net:
    ROTATION = np.array([[0, -1], [1, 0]])
    def __init__(self, numNeurons, refractoryPeriodMean=4, refractoryPeriodSigma=3, historyLength=700):
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
        self.baseThresholds = np.ones(self.numNeurons)
        self.thresholds = np.ones(self.numNeurons)
        self.refractoryPeriods = np.round(np.random.normal(loc=refractoryPeriodMean, scale=refractoryPeriodSigma, size=self.numNeurons))
        self.refractoryCountdowns = np.zeros(self.numNeurons)
        self.connectionArrows = [[None for k in range(self.numNeurons)] for j in range(self.numNeurons)]
        self.neuronMarkers = [None for k in range(self.numNeurons)]
        # Neurons can be given custom attributes. Each attribute has an entry
        #   in the attributeNames array, and a corresponding row in the
        #   attribute values matrix. Attribute values must be numerical.
        self.attributeNames = []
        self.attributeValues = np.zeros([0, self.numNeurons])

    def addAttributes(self, name, indices=None, initialValues=None):
        # name = string, name of new attribute
        # initialValues = list of initial values, or None to initialize with all
        #   zeros. If a list, it must have length equal to numNeurons
        if name in self.attributeNames:
            KeyError('Attribute "{n}" already exists.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
        self.attributeNames.append(name)
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
        np.delete(self.attributeValues, idx, 0)

    def getAttributes(self, name, indices=None):
        # Return a value or array of values corresponding to the attribute
        #   specified by name and the neuron indices specified by indices.
        #   If indices is None, this returns attribute values for all neurons.
        try:
            idx = self.attributeNames.index(name)
        except ValueError():
            raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
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

    def setAttributes(self, name, values=0, indices=None, addIfNonexistent=True):
        # Set a value or array of values corresponding to the attribute
        #   specified by name and the neuron indices specified by indices. The
        #   array of values must be the same size as the array of indices.
        #   If indices is None, this sets attribute values for all neurons.
        try:
            idx = self.attributeNames.index(name)
        except ValueError:
            if addIfNonexistent:
                idx = self.addAttributes(name)
            else:
                raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
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

    def randomizeConnections(self, n, mu, sigma, indicesA=None, indicesB=None, attributeName=None, attributeValueA=None, attributeValueB=None, modulatory=False):
        # Change random connections
        #   n = number of connections to change
        #   mu = mean connection strength
        #   sigma = standard deviation of connection strength
        #   indicesA = the indices of neurons to choose from to randomly connect
        #       from. If None, all neurons may be chosen to connect from.
        #   indicesB = the indices of neurons to choose from to randomly connect
        #       to. If None, all neurons may be chosen to connect to.
        #   attributeName = an alternate way to select neurons, by attribute name
        #   attributeValueA = the value of the indicated attribute to select
        #       neurons to connect from.
        #   attributeValueB = the value of the indicated attribute to select
        #       neurons to connect to. If None, the same group of neurons
        #       are selected for connections from and to.
        #   modulatory = boolean flag indicating that the modulatory network
        #       instead of the direct network should be randomized.

        if attributeValueB is None:
            attributeValueB = attributeValueA
        if indicesB is None:
            indicesB = indicesA
        indicesA = self.getIndices(indices=indicesA, attributeName=attributeName, attributeValue=attributeValueA)
        indicesB = self.getIndices(indices=indicesB, attributeName=attributeName, attributeValue=attributeValueB)

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

    def createChain(self):
        for k in range(self.numNeurons-1):
            n.connections[k, k+1] = 10

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
    N = 1000
    NL = N//50
    n = Net(N, refractoryPeriodMean=10, refractoryPeriodSigma=7)
    regularIndices = np.arange(0, 800)
    modulatoryIndices = np.arange(800, N)
    n.createLayers(nLayers=NL,  nConnectionsPerLayer=N, nInterconnects=N//10, mu=0.5, sigma=2, indices=regularIndices)
    n.randomizeConnections(N*10, 0, 20, indicesA=modulatoryIndices, indicesB=regularIndices, modulatory=True)
    n.randomizeConnections(N*10, 0, 20, indicesA=regularIndices, indicesB=modulatoryIndices, modulatory=True)
    #     n.randomizeConnections(N, 0.5, 2, attribute='layer', attributeValueA=0, attributeValueB=layerNum)

    # layerNums = n.getUniqueAttributes('layer')
    # for layerNum in layerNums:
#    n.createChain()

#    n.randomizeConnections(100, 2, 1)

    # n.randomizeConnections(N*N, 0, 2)
#    n.arrangeNeurons()
#    n.showNet()
    stim = np.zeros(N)
    stim[0:10] = 10
    n.addInput(stim)
    n.run(n.historyLength)
    f, axs = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [3, 1]})
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
    cRadius = max(np.max(n.connections), abs(np.min(n.connections)))
    connIm.set_clim(-cRadius, cRadius)
    axs[0, 1].set_ylabel("Upstream neuron #")
    axs[0, 1].set_xlabel("Downstream neuron #")

    layerNums = n.getUniqueAttributes('layer')
    for layerNum in layerNums:
        layerIdx = n.filterByAttribute('layer', layerNum)
        if len(layerIdx) == 0:
            continue
        axs[1, 0].plot(10*np.mean(np.flip(n.history[layerIdx, :], axis=1), axis=0) + (max(layerNums) - layerNum)*2)

    plt.show()
