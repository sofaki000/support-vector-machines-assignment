import numpy as np
from matplotlib import pyplot as plt

def GRBF(x1, x2):
    diff = x1 - x2
    return np.exp(-np.dot(diff, diff) * len(x1) / 2)

def linear_kernel(x, y):
    return np.dot(x, y.T)

def polynomial(x,y,degree):
		return (np.dot(x.T,y))**degree

def rbf(x,y,gamma):
	return np.exp(-1.0*gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))


# metatrepei mia seira X me ton kernel pou dinetai.
def transform(X):
    K = np.zeros([X.shape[0], X.shape[0]])
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i, j] = linear_kernel(X[i], X[j])
    return K

x = [1,2,3,4,5,6] #np.linspace(-10,10)
y = np.linspace(-10,10)
ks = []

# for i in range(len(x)):
#     ks.append(linear_kernel(i))

plt.plot(x)
plt.plot(transform(x))
plt.show()