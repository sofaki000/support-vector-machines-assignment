import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from data_utilities.test_data import generateBatchBipolar, generateBatchXor
from model import run_sklearn_svm
from plot_utilities import plotSvm

N = 100

X_train, y_train = generateBatchXor(n=N)
X_test, y_test = generateBatchXor(n=N)
# X_train, y_train = generateBatchBipolar(N,  mu=0.3, sigma=0.2)
# X_test, y_test = generateBatchBipolar(N,  mu=0.3, sigma=0.2)

# C controls the trade off between smooth decision boundary and
# classifying training points correctly. A large value of c means you will get more training points correctly.
# the bigger the C, the more it will create complicated curves to fit everything
# for C in [0.001, 0.0001 ]:
#     run_sklearn_svm(X_train, X_test,y_train,y_test, C=C, file_name=f"results\svm_xor_{C}")


# kernels

def run_svm_with_kernel(C=10, gamma=1/2):
    classifier = SVC(kernel='rbf', C=10, gamma=gamma, shrinking=False)
    classifier.fit(X_train, y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = float(cm.diagonal().sum()) / len(y_test)

    fig, ax = plt.subplots(1, figsize=(11, 7))


    plotSvm(X_train, y_train, support=classifier.support_vectors_, label='Training', ax=ax)
    acc_result = f"\nAccuracy: {accuracy},   C={C}"
    print(acc_result)
    plt.title(acc_result, fontsize=8)
    # Estimate and plot decision boundary
    xx = np.linspace(-1, 1, 50)
    X0, X1 = np.meshgrid(xx, xx)
    xy = np.vstack([X0.ravel(), X1.ravel()]).T

    # Estimate and plot decision boundary
    Y31 = classifier.predict(xy).reshape(X0.shape)
    ax.contour(X0, X1, Y31, colors='k', levels=[-1, 0], alpha=0.3, linestyles=['-.', '-'])
    plt.savefig(f"results/rbf/svc_rbf_c{C}g{gamma}.png")


gamma_2d_range = [1e-1, 1, 1e1, 1e2, 1e3]
for gamma in gamma_2d_range:
    run_svm_with_kernel(gamma=gamma)