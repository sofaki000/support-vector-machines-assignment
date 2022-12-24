import numpy as np
import matplotlib.pyplot as plt

from data_utilities.test_data import get_uniform_data, get_labels_for_uniform_data
from svm import SVM

np.random.seed(1)


def main():
    N = 100
    data = get_uniform_data(0, 1, N, 2)
    labels = get_labels_for_uniform_data(data).reshape(-1)
    predictions = np.ones_like(labels) * -1
    print("Max-class classifier training set accuracy: ", np.mean(np.equal(predictions, labels)) * 100, "%")
    model = SVM(kernel="rbf", gamma=3)
    model.fit(data, labels)
    predictions = model.predict(data)
    print("SVM model Training set accuracy: ", np.mean(np.equal(predictions, labels)) * 100, "%")
    print("Number of SVMs computed: ", model._n_support)
    print("Value of intercept: ", model.intercept)
    # print(model.project(data))

    color = np.where(model._support_labels == 1, "orange", "green") #an einai support vector tou 1 class
    # balto sto orange, an einai sv tou -1 class balto green

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap=plt.cm.bwr)

    plt.scatter(model._support_vectors[:, 0], model._support_vectors[:, 1], s=35, c=color, marker='H')
    plt.title('SVM Boundaries (N = %d)' % (N))

    X1, X2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X_T = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    #
    Z = model.project(X_T).reshape(X1.shape)
    #
    H = plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=2, origin='lower')
    # H_1 = plt.contour(X1, X2, Z + 1, [0.0], colors='tab:green', linewidths=1, origin='lower')
    # H1 = plt.contour(X1, X2, Z - 1, [0.0], colors='orange', linewidths=1, origin='lower')
    #
    # # -1, 0, +1
    # plt.clabel(H, inline=True, fmt="0", fontsize=8)
    # plt.clabel(H_1, inline=True, fmt="-1", fontsize=8)
    # plt.clabel(H1, inline=True, fmt="+1", fontsize=8)
    plt.axis("tight")
    plt.show()


if __name__ == '__main__':
    main()