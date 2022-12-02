import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from data_utilities.test_data import get_test_data
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_train, X_test,y_train,y_test = get_test_data()

def run_sklearn_svm(X_train, X_test,y_train,y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    classifier = SVC(kernel='linear', random_state = 1)
    classifier.fit(X_train,y_train)

    Y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test,Y_pred)

    accuracy = float(cm.diagonal().sum())/len(y_test)

    print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

    # plot_confusion_matrix(classifier, X_test, y_test)
    # plt.show()

    plt.figure(figsize=(10, 8))
    # Plotting our two-features-space
    sns.scatterplot(x=X_train[:, 0],  y=X_train[:, 1],  hue=y_train, s=8);
    # Constructing a hyperplane using a formula.
    w = classifier.coef_[0]           # w consists of 2 elements
    b = classifier.intercept_[0]      # b consists of 1 element
    x_points = np.linspace(-10, 10)    # generating x-points from -10 to 10
    y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

    # Constructing a hyperplane using a formula.
    y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
    # Plotting a red hyperplane
    plt.plot(x_points, y_points, c='r');
    #  Encircle support vectors
    plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=50, facecolors='none', edgecolors='k', alpha=.5);
    # Step 2 (unit-vector):
    w_hat = w / (np.sqrt(np.sum(w ** 2)))
    # Step 3 (margin):
    margin = 1 / np.sqrt(np.sum(w ** 2))
    # Step 4 (calculate points of the margin lines):
    decision_boundary_points = np.array(list(zip(x_points, y_points)))
    points_of_line_above = decision_boundary_points + w_hat * margin
    points_of_line_below = decision_boundary_points - w_hat * margin
    # Plot margin lines
    # Blue margin line above
    plt.plot(points_of_line_above[:, 0], points_of_line_above[:, 1], 'b--',   linewidth=2)
    # Green margin line below
    plt.plot(points_of_line_below[:, 0],  points_of_line_below[:, 1], 'g--', linewidth=2)

    plt.savefig("sklearn_algorithm.png")