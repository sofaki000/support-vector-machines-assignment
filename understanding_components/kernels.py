import numpy as np
import seaborn as sns
from sklearn.datasets import make_circles

sns.set()

def feature_map_1(X):
    return np.asarray((X[:,0], X[:,1], X[:,0]**2 + X[:,1]**2)).T
def feature_map_2(X):
    return np.asarray((X[:,0], X[:,1], np.exp( -( X[:,0]**2 + X[:,1]**2)))).T
def feature_map_3(X):
    return np.asarray(( np.sqrt(2) *X[:,0] * X[:,1], X[:,0]**2, X[:,1]**2)).T

def my_kernel_1(X,Y):
    x = feature_map_1(X)
    x_t  = feature_map_1(Y).T
    return np.dot(x, x_t)

def my_kernel_2(X,Y):
    return np.dot(feature_map_2(X),feature_map_2(Y).T )

def my_kernel_3(X,Y):
    return np.dot(feature_map_3(X),feature_map_3(Y).T )

#Generate dataset and feature-map
X, y = make_circles(100, factor=.1, noise=.3, random_state = 0)
