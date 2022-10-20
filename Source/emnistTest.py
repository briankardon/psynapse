import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import net
import netEvolve as ne
from PIL import Image
import sys

TRANSPOSE = Image.Transpose.TRANSPOSE
letters = 'abcdefghijklmnopqrstuvwxyz'

def showImage(k, images, labels):
    letter = letters[labels[k]-1]
    Image.fromarray(images[k, :].reshape([28, 28])).transpose(method=TRANSPOSE).show(title=letter)

# EMNIST datset:
# Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an
#   extension of MNIST to handwritten letters. Retrieved from
#   http://arxiv.org/abs/1702.05373

if __name__ == '__main__':
    dataset = scipy.io.loadmat('..\Resources\emnist-letters.mat')['dataset']

    images = dataset['train'][0, 0]['images'][0, 0]
    labels = dataset['train'][0, 0]['labels'][0, 0][:, 0]
    numLabels = len(np.unique(labels))
    # We're depending on the fact that the labels are consecutive integers starting at 0
    numImages = images.shape[0]
    imageSize = images.shape[1]

    # Pregenerate reusable target arrays, to save memory
    # Each targets array is a label vector with 1 for the correct label, 0 for all others
    targets = []
    for k in range(numLabels):
        newTarget = np.zeros(numLabels)
        newTarget[k] = 1;
        targets.append(newTarget)

    print('Number of labels: {nL}'.format(nL=numLabels))
    print('Number of images: {nI}'.format(nI=numImages))

    numFeedbackNeurons = 10
    interactors = []
    trainTime = 500
    numInputNeurons = imageSize + numFeedbackNeurons

    numTrainingImages = 1000
    trainingImageIndices = list(range(numTrainingImages))

    debugMode = True

    if debugMode:
        populationSize=10
        nGens=1
        netsPerConnectome=2
        testsPerNet=3
        numWorkers=None
    else:
        populationSize=100
        nGens=10
        netsPerConnectome=10
        testsPerNet=5
        numWorkers=8
    saveDir = r'D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\psynapse\Source\evolveSessions\emnist_learner'
    if len(sys.argv) < 2:
        seedPath = 'letterRecognitionSeedConnectome.csv'
    else:
        seedPath = sys.argv[1]
    if len(sys.argv) < 3:
        mode = 'evolve'
    else:
        mode = sys.argv[2]

    print('Using seed {s}'.format(s=seedPath))

    for k in trainingImageIndices:
        # Stimulus is a flattened image 1D array
        stimulus = images[k, :]
        target = targets[labels[k]-1]
        ti = ne.TonicInteractor(stimulus, trainTime, targets=target, averagingTime=5)
        fi = ne.FeedbackInteractor(ti, numFeedbackNeurons, feedbackTimeSteps=None)
        interactors.append(fi)

        # plt.imshow(np.transpose(images[k, :].reshape([28, 28])))
        # plt.gca().title.set_text(str(labels[k]))
        # plt.show()

    if mode == 'evolve':
        seedConnectomes = [net.Connectome(seedPath)]
        print('pops: ', seedConnectomes[0].populations)
        ce = ne.ConnectomeEvolver(seedConnectomes, interactors, populationSize=populationSize, inputRegion='I', outputRegion='O')
        ce.evolve(nGens=nGens, nNets=netsPerConnectome, testsPerNet=testsPerNet, saveDir=saveDir, saveName='emnist', numWorkers=numWorkers, saveAllGenerations=True)
    elif mode == 'demo':
        print('Loading connectome')
        connectome = net.Connectome(seedPath);
        print('Creating net')
        n = connectome.createNet()
        print('Choosing interactor')
        idx = np.random.choice(trainingImageIndices, size=1)[0]
        print('Chose #', idx)
        ia = interactors[idx]
        print('Running interactor...')
        output = n.runInteraction(ia,
            inputAttributeName='region',
            inputAttributeValue='I',
            inputMapped=True,
            outputAttributeName='region',
            outputAttributeValue='O',
            outputMapped=True)
        print('...done running interactor')
        print('Target:     ', ia.getTargets())
        print('Firing rate:', ia.getFiringRate())
        score = ia.scoreOutput()
        print('Score:      ', score)
        showImage(idx, images, labels)



    # images[k, :]
    #
    # ce = ConnectomeEvolver(['TestConnectome'], )
