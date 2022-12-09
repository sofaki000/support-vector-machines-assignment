from sklearn.datasets import make_blobs
import numpy as np

from plot_utilities import plotSvm


def get_test_data(num_samples=1000, train_percentage=0.8):

    indx_split = int(num_samples * train_percentage)
    dataX, datay = make_blobs(n_samples=num_samples, centers=2, n_features=2, cluster_std=2, random_state=2)
    X, x_test = dataX[:indx_split], dataX[indx_split:]
    y, y_test = datay[:indx_split], datay[indx_split:]
    return X, x_test, y,y_test

def generateBatchXor(n, mu=0.5, sigma=0.5):
    """ Four gaussian clouds in a Xor fashion """
    X = np.random.normal(mu, sigma, (n, 2))
    yB0 = np.random.uniform(0, 1, n) > 0.5
    yB1 = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y0 = 2. * yB0 - 1
    y1 = 2. * yB1 - 1
    X[:,0] *= y0
    X[:,1] *= y1
    X -= X.mean(axis=0)
    return X, y0*y1

# N = 100
# xTrain3, yTrain3 = generateBatchXor(2*N, sigma=0.25)
# plotSvm(xTrain3, yTrain3)
# xTest3, yTest3 = generateBatchXor(2*N, sigma=0.25)

def generateBatchBipolar(n, mu=0.5, sigma=0.2):
    """ Two gaussian clouds on each side of the origin """
    X = np.random.normal(mu, sigma, (n, 2))
    yB = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y = 2. * yB - 1
    X *= y[:, np.newaxis]
    X -= X.mean(axis=0)
    return X, y


# N = 100
# xTrain0, yTrain0 = generateBatchBipolar(N,  mu=0.5, sigma=0.2)
# plotSvm(xTrain0, yTrain0)