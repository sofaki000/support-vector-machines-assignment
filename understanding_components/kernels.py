import numpy as np
import seaborn as sns
from sklearn.datasets import make_circles

sns.set()

# an to X einai array me polla points:
def feature_map_1(X):
    return np.asarray((X[:,0], X[:,1], X[:,0]**2 + X[:,1]**2)).T

def feature_map_2(X):
    return np.asarray((X[:,0], X[:,1], np.exp( -( X[:,0]**2 + X[:,1]**2)))).T

def feature_map_3(X):
    return np.asarray(( np.sqrt(2) *X[:,0] * X[:,1], X[:,0]**2, X[:,1]**2)).T


# An to X to predict einai ena mono point:
def feature_map_1_x_point(X):
    return np.asarray((X[0], X[1], X[0] ** 2 + X[1] ** 2)).T

def my_kernel_1(support_vectors,Y, is_x_point=False):
    if True: #is_x_point is False:
        x = feature_map_1(support_vectors)
        x_t = feature_map_1(Y).T

        # to apotelesma prepei na einai n*n pinakas an tou dinoume oloklhro array
        return np.dot(x, x_t)
    else:
        x = feature_map_1(support_vectors)
        x_t = feature_map_1_x_point(Y).T

        # to apotelesma prepei na einai arithmos an tou dinoume 1 point mono
        return np.sum(np.dot(x, x_t))


def my_kernel_2(X,Y, is_x_point=False):
    return np.dot(feature_map_2(X),feature_map_2(Y).T )

def my_kernel_3(X,Y, is_x_point=False):
    return np.dot(feature_map_3(X),feature_map_3(Y).T )

#Generate dataset and feature-map
X, y = make_circles(100, factor=.1, noise=.3, random_state = 0)
