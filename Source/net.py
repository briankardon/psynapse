import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class Net:
    rotation = np.array([[0, -1], [1, 0]])

    def __init__(self, numNeurons, connector=None):
        self.numNeurons = numNeurons
        self.connections = np.zeros([self.numNeurons, self.numNeurons])
        self.neurons = [Neuron(self, k) for k in range(self.numNeurons)]
        self.activations = np.zeros(self.numNeurons)
        self.thresholds = np.ones(self.numNeurons)
        self.connectionArrows = [[None for k in range(self.numNeurons)] for j in range(self.numNeurons)]
        self.neuronMarkers = [None for k in range(self.numNeurons)]
        # Neurons can be given custom attributes. Each attribute has an entry
        #   in the attributeNames array, and a corresponding row in the
        #   attribute values matrix. Attribute values must be numerical.
        self.attributeNames = []
        self.attributeValues = np.zeros([0, self.numNeurons])

    def addAttributes(name, initialValues=None):
        # name = string, name of new attribute
        # initialValues = list of initial values, or None to initialize with all
        #   zeros. If a list, it must have length equal to numNeurons
        if name in self.attributeNames:
            KeyError('Attribute "{n}" already exists.'.format(n=name))
        self.attributeNames.append(name)
        self.attributeValues = self.attributeValues.pad(self.attributeValues, ([0, 1], [0, 0]))
        if initialValues is not None:
            self.attributeValues[-1, :] = initialValues

    def removeAttributes(name):
        # Deletes the specified attribute.
        try:
            idx = self.attributeNames.index(name)
        except ValueError():
            raise KeyError('Attribute "{n}" does not exist.'.format(n=name))
        self.attributeNames.pop(idx)
        np.delete(self.attributeValues, idx, 0)

    def getAttributes(name, indices=None):
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

    def filterByAttribute(name, value):
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

    def getUniqueAttributes(name):
        return np.unique(self.getAttributes(name))

    def setAttributes(name, values, indices=None, addIfNonexistent=True):
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
        self.activations = (self.activations > self.thresholds).dot(self.connections)

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

    def randomizeConnections(self, n, mu, sigma, indices=None):
        # Change random connections
        #   n = number of connections to change
        #   mu = mean connection strength
        #   sigma = standard deviation of connection strength
        #   indices = the indices of neurons to choose from to randomly connect.
        #       If None, all neurons may be chosen to connect

        if indices is None:
            indices = np.arange(self.numNeurons)

        # Get random neuron coordinates
        x = np.random.choice(indices, size=n, replace=True)
        y = np.random.choice(indices, size=n, replace=True)
        # Get random connection strengths
        c = np.random.normal(loc=mu, scale=sigma, size=n)
        # Change random connections
        self.connections[x, y] = c

    # def createLayers(self, nLayers):
    #     layerNumbers = [np.floor(x/(self.numNeurons/nLayers)) for x in range(self.numNeurons)]
    #     self.addAttributes('layer', layerNumbers)
    #     layerNums = self.getUniqueAttributes('layer')
    #     for layerNum in layerNums:
    #         self.

    def downRegulateAutapses(self, factor):
        # Reduce the strength of autapses
        #   factor = the factor by which the mean autapse strength should be
        #       smaller than the mean synapse strength
        x = range(self.numNeurons)

    def arrangeNeurons(self):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.connections)
        for k in range(self.numNeurons):
            self.neurons[k].setPosition(principalComponents[k, :])

    def showNet(self):
        x = [nr.getX() for nr in self.neurons]
        y = [nr.getY() for nr in self.neurons]
        maxC = np.absolute(self.connections).max()
        markerSize = 10
        markerRadius = np.sqrt(markerSize)/20
        plt.axes().set_aspect(1)
        plt.scatter(x, y, s=markerSize)
        arrowStyle = "Simple, tail_width=0.5, head_width=4, head_length=8"
        for k in range(self.numNeurons):
            for j in range(self.numNeurons):
                p1 = np.array([x[k], y[k]])
                p2 = np.array([x[j], y[j]])
                dir = p2 - p1
                dirHat = dir / np.linalg.norm(dir)
                dir2Hat = dirHat.dot(Net.rotation)
                sideOffset = markerRadius * dir2Hat
                forwardOffset = markerRadius * dirHat
                p1 = p1 + sideOffset + forwardOffset
                p2 = p2 + sideOffset - forwardOffset
                c = self.connections[k][j] / maxC
                if c > 0:
                    color = np.array([0, 1, 0])
                else:
                    color = np.array([1, 0, 0])
                brightness = abs(c)
                if k == j:
                    # Autapse
                    self.connectionArrows[k][j] = patches.FancyArrowPatch(p1, p2, arrowstyle=arrowStyle, color=color, alpha=brightness, connectionstyle="arc3,rad={r}".format(r=markerRadius*5))
#                    self.connectionArrows[k][j] = plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], color=color, alpha=brightness, head_width=0.1)
                else:
                    # Synapse
                    self.connectionArrows[k][j] = patches.FancyArrowPatch(p1, p2, arrowstyle=arrowStyle, color=color, alpha=brightness)
#                    self.connectionArrows[k][j] = plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], color=color, alpha=brightness, head_width=0.1)
                plt.gca().add_patch(self.connectionArrows[k][j])
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
                print(self.activations)

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
    N = 10
    n = Net(N)
    n.randomizeConnections(N*N, 0, 2)
    print(n.connections)
    n.arrangeNeurons()
    n.showNet()
    n.addInput(10*np.ones(N))
    print('activations:')
    print(n.activations)
    print('thresholds')
    print(n.thresholds)
    n.run(20)
