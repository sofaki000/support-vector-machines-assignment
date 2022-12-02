import numpy as np
from matplotlib import pyplot as plt
from data_utilities.test_data import get_test_data
from model import run_sklearn_svm

epochs = 10000
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=epochs):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    def get_svm_lines(self):
        w1 = self.w[0]
        w2 = self.w[1]
        b = self.b
        w_hat = w1 / np.sqrt(np.sum(w1 ** 2))

        margin = 1 / np.sqrt(np.sum(w1 ** 2))

        def boundary_line(points):
            # return np.dot(points, self.w) + self.b
            return np.array(points)* -(w2/w1)- b*w2

        def up_line(hyperplane):
            return hyperplane + w_hat * margin
        def down_line(hyperplane):
            return hyperplane - w_hat * margin

        return boundary_line, up_line, down_line
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)



svm = SVM()
x_train, x_test, y_train, y_test =get_test_data(num_samples=100)
svm.fit(x_train, y_train)

correct = 0
for i in range(len(x_test)):
    prediction = svm.predict(x_test[i])
    if prediction==y_test[i]:
        correct+=1

acc = 100 * correct/len(x_test)

plt.scatter(x_test[:, 0], x_test[:, 1])
one_class_xs = []
one_class_ys = []
other_class_xs = []
other_class_ys = []
for i in range(len(x_test)):
    if y_test[i]==1:
        one_class_xs.append(x_test[i, 0])
        one_class_ys.append(x_test[i, 1])
    else:
        other_class_xs.append(x_test[i, 0])
        other_class_ys.append(x_test[i, 1])
for i in range(len(x_train)):
    if y_train[i]==1:
        one_class_xs.append(x_train[i, 0])
        one_class_ys.append(x_train[i, 1])
    else:
        other_class_xs.append(x_train[i, 0])
        other_class_ys.append(x_train[i, 1])

plt.scatter(one_class_xs, one_class_ys, color="red", s=50, facecolors='none', edgecolors='k', alpha=.5)
plt.scatter(other_class_xs, other_class_ys, color="blue", s=50, facecolors='none', edgecolors='k', alpha=.5)
plt.title(f'Accuracy:{acc}%')

boundary_line, up_line, down_line = svm.get_svm_lines()
x = np.linspace(-10, 10, 2)
hyperplane = boundary_line(x)
plt.plot(x, hyperplane, color='red')
plt.plot(x, up_line(hyperplane), color='green')
plt.plot(x, down_line(hyperplane), color='blue')
plt.savefig("svm_handwritten.png")
print(f'Accuracy:{acc}%')

run_sklearn_svm(x_train, x_test,y_train,y_test)