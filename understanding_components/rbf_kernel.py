import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.1)
# X = (X - X.mean()) / X.std()
# y[np.where(y == 0)] = -1
#
# clf = SVC(kernel='rbf')
# clf.fit(X, y)
# # the non-zero alphas antistoixoun sta support vectors (an ena point den einai support vector, exei a=0)
# support_vectors_number= np.abs(clf.dual_coef_).shape[1]
# sv_num_content = f'Support vectors num:{support_vectors_number}\n'
#

xk = np.linspace(0,1,5)
x = np.linspace(0,1,100)
def true_fn(x):
    return x**2-x-np.cos(np.pi*x)

# plt.figure(figsize=(12,6))
# plt.plot(xk, true_fn(xk), 'x', markersize=15)
# plt.plot(x, true_fn(x), '--r')
# plt.show()


def euclidean_distance(x, xk):
    distances_between_points_with_each_other = np.sqrt((x.reshape(-1, 1) - xk.reshape(1, -1)) ** 2)
    return distances_between_points_with_each_other

print(euclidean_distance(xk,xk))

def gauss_rbf(radius, epsilon):
    return np.exp(-(epsilon*radius)**2)

print(gauss_rbf(euclidean_distance(xk,xk), 2))

class RBFInterpolation(object):
    def __init__(self, eps):
        self.eps = eps
    def fit(self,xk,yk):
        self.xk = xk
        transformation = gauss_rbf(euclidean_distance(xk,xk), self.eps)
        self.w = np.linalg.solve(transformation, yk)
    def __call__(self, xn):
        transformation = gauss_rbf(euclidean_distance(xn,self.xk), self.eps)
        return transformation.dot(self.w)

    def get_influence_of_xn_to_point(self, points):
        transformation = gauss_rbf(euclidean_distance(points, self.xk[0]), self.eps)
        import matplotlib.pyplot as plt

        x = np.linspace(-3, 3, num=100)

        plt.figure(figsize=(12, 6))
        plt.plot(points, transformation.reshape(-1),  markersize=15)
        plt.plot(x, transformation,  'x', markersize=15)

        plt.show()



yk = true_fn(xk)
model = RBFInterpolation(eps=2)
model.fit(xk,yk)
model.get_influence_of_xn_to_point(x)

predictions = model(x)

# plt.figure(figsize=(12,6))
# plt.plot(xk,yk, 'x')
# plt.plot(x, predictions,'--r')
# # plotting real function
# plt.plot(x, true_fn(x), 'b')
# plt.show()



def np_bivariate_normal_pdf(domain, mean, variance):
  X = np.arange(-domain+mean, domain+mean, variance)
  Y = np.arange(-domain+mean, domain+mean, variance)
  X, Y = np.meshgrid(X, Y)
  R = np.sqrt(X**2 + Y**2)
  Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))
  return X+mean, Y+mean, Z


def plt_plot_bivariate_normal_pdf(x, y, z, name):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca(projection='3d')
    # ax.plot_surface(x, y, z,
    #                 cmap=cm.coolwarm,
    #                 linewidth=0,
    #                 antialiased=True)
    ax.plot_surface(x, y, z,
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()