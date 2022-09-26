import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import net
import netEvolve as ne
from PIL import Image

TRANSPOSE = Image.Transpose.TRANSPOSE
letters = 'abcdefghijklmnopqrstuvwxyz'

def showImage(k, images, labels):
    letter = letters[labels[k][0]-1]
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
    trainTime = 100
    numInputNeurons = imageSize + numFeedbackNeurons

    numTrainingImages = 1000
    trainingImageIndices = list(range(numTrainingImages))

    populationSize=10
    nGens=1
    netsPerConnectome=2
    testsPerNet=1
    numWorkers=None
    saveDir = r'D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\psynapse\Source\evolveSessions\emnist_learner'

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

    seedConnectomes = [net.Connectome('letterRecognitionSeedConnectome.csv')]
    print('pops: ', seedConnectomes[0].populations)
    ce = ne.ConnectomeEvolver(seedConnectomes, interactors, populationSize=populationSize, inputRegion='I', outputRegion='O')
    ce.evolve(nGens=nGens, nNets=netsPerConnectome, testsPerNet=testsPerNet, saveDir=saveDir, saveName='emnist', numWorkers=numWorkers)


    # images[k, :]
    #
    # ce = ConnectomeEvolver(['TestConnectome'], )
