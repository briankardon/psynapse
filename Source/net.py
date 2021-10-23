import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class Net:
    rotation = np.array([[0, -1], [1, 0]])

    def __init__(self, numNeurons, connector=None):
        self.historyLength = 1000
        self.numNeurons = numNeurons
        self.connections = np.zeros([self.numNeurons, self.numNeurons])
        self.neurons = [Neuron(self, k) for k in range(self.numNeurons)]
        self.activations = np.zeros(self.numNeurons)
        self.history = np.zeros([self.numNeurons, self.historyLength])
        self.thresholds = np.ones(self.numNeurons)
        self.refractoryPeriods = np.round(np.random.normal(loc=4, scale=3, size=self.numNeurons))
        self.refractoryCountdowns = np.zeros(self.numNeurons)
        self.connectionArrows = [[None for k in range(self.numNeurons)] for j in range(self.numNeurons)]
        self.neuronMarkers = [None for k in range(self.numNeurons)]
        # Neurons can be given custom attributes. Each attribute has an entry
        #   in the attributeNames array, and a corresponding row in the
        #   attribute values matrix. Attribute values must be numerical.
        self.attributeNames = []
        self.attributeValues = np.zeros([0, self.numNeurons])

    def addAttributes(self, name, initialValues=None):
        # name = string, name of new attribute
        # initialValues = list of initial values, or None to initialize with all
        #   zeros. If a list, it must have length equal to numNeurons
        if name in self.attributeNames:
            KeyError('Attribute "{n}" already exists.'.format(n=name))
        self.attributeNames.append(name)
        self.attributeValues = np.pad(self.attributeValues, ([0, 1], [0, 0]))
        if initialValues is not None:
            self.attributeValues[-1, :] = initialValues

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
            idx = np.arange(self.numNeurons)[values==value]
        return idx

    def getUniqueAttributes(self, name):
        return np.unique(self.getAttributes(name))

    def setAttributes(self, name, values, indices=None, addIfNonexistent=True):
        # Set a value or array of values corresponding to the attribute
        #   specified by name and the neuron indices specified by indices. The
        #   array of values must be the same size as the array of indices.
        #   If indices is None, this sets attribute values for all neurons.
        try:
            idx = self.attributeNames.index(name)
        except ValueError():
            if addIfNonexistent:
                self.addAttributes(name)
            else:
                raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        if indices is None:
            indices = np.s_[:]
        self.attributeValues[idx, indices] = values

    def activate(self):
        # Simulate activation of neurons and transmission of action potentials
        self.history = np.roll(self.history, 1, axis=1)
        firing = (self.refractoryCountdowns <= 0) & (self.activations > self.thresholds)
        self.activations = firing.dot(self.connections)
        # Reset fired neurons to max countdown, continuing decrementing the rest.
        self.refractoryCountdowns = (self.refractoryPeriods * firing) + ((self.refractoryCountdowns - 1) * (np.logical_not(firing)))
        # print('ACTIVATE:')
        # print('firing')
        # print(firing * 1.0)
        # print('periods')
        # print(self.refractoryPeriods)
        # print('countdowns:')
        # print(self.refractoryCountdowns)
        self.history[:, 0] = firing

    def addInput(self, inputs, indices=None):
        # Set activations of neurons indicated by indices to the values in inputs
        #   inputs must be an iterables of less than or equal length to
        #       numNeurons indicating how much to add to the neuron's activation
        #       input
        #   indices must be either an iterable of the same length as inputs
        #       indicating which neurons to activate, or None, indicating the
        #       inputs should just be applied to the first N neurons
        if indices is None:
            indices = np.arange(len(inputs))
        self.activations[indices] = inputs

    def randomizeConnections(self, n, mu, sigma, indicesA=None, indicesB=None):
        # Change random connections
        #   n = number of connections to change
        #   mu = mean connection strength
        #   sigma = standard deviation of connection strength
        #   indicesA = the indices of neurons to choose from to randomly connect
        #       from. If None, all neurons may be chosen to connect from.
        #   indicesB = the indices of neurons to choose from to randomly connect
        #       to. If None, all neurons may be chosen to connect to.

        if indicesA is None:
            indicesA = np.arange(self.numNeurons)
        if indicesB is None:
            indicesB = indicesA

        # Get random neuron coordinates
        x = np.random.choice(indicesA, size=n, replace=True)
        y = np.random.choice(indicesB, size=n, replace=True)
        # Get random connection strengths
        c = np.random.normal(loc=mu, scale=sigma, size=n)
        # Change random connections
        self.connections[x, y] = c

    def createLayers(self, nLayers, nConnectionsPerLayer, nInterconnects, mu, sigma):
        layerNumbers = np.array([np.floor(x/(self.numNeurons/nLayers)) for x in range(self.numNeurons)])
        self.addAttributes('layer', layerNumbers)
        layerNums = self.getUniqueAttributes('layer')
        for k, layerNum in enumerate(layerNums):
            # Make within-layer connections
            layerIndices = np.arange(self.numNeurons)[layerNumbers==layerNum]
            self.randomizeConnections(nConnectionsPerLayer, mu, sigma, indicesA=layerIndices)
            if k < len(layerNums)-1:
                # Make interconnects
                nextLayerIndices = np.arange(self.numNeurons)[layerNumbers==layerNums[k+1]]
                self.randomizeConnections(nInterconnects, mu, sigma, indicesA=layerIndices, indicesB=nextLayerIndices)

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
                    dir2Hat = dirHat.dot(Net.rotation)
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
                    dir2Hat = dirHat.dot(Net.rotation)
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
#                print(n.history)

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
    NL = N//100
    n = Net(N)
#    n.createLayers(nLayers=4,  nConnectionsPerLayer=N, nInterconnects=N//2, mu=0, sigma=2)
    n.createLayers(nLayers=NL,  nConnectionsPerLayer=N, nInterconnects=N//2, mu=0, sigma=2)
#    n.randomizeConnections(100, 2, 1)

    # n.randomizeConnections(N*N, 0, 2)
    print(n.connections)
#    n.arrangeNeurons()
    print([nr.position for nr in n.neurons])
#    n.showNet()
    stim = np.zeros(N)
    stim[1:(N//NL)] = 10
    n.addInput(stim)
    n.run(n.historyLength)
    plt.imshow(n.history, cmap='gist_gray') #, 'XData', np.arange(n.historyLength))
    # for k in range(n.numNeurons):
    #     plt.step(np.arange(n.historyLength), 5*n.history[k, :] + 10*k)
    # for k in range(n.numNeurons):
    #     for j in range(n.numNeurons):
    #         print('{c:0.01f} '.format(c=n.connections[k][j]), end='')
    #     print()
    plt.show()
