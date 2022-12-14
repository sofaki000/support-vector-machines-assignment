import numpy as np
from sklearn.datasets import make_circles, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from data_utilities.test_data import generateBatchXor
from understanding_components.kernels import my_kernel_1, my_kernel_2, my_kernel_3

# X, y = make_circles(100, factor=.1, noise=.3, random_state = 0)
# y[np.where(y == 0)] = -1
#
X, y = make_blobs(n_samples=400, centers=2, random_state=0, cluster_std=0.5)
X = (X - X.mean()) / X.std()
y[np.where(y == 0)] = -1

X_test, y_test = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.1)
y_test[np.where(y_test == 0)] = -1
# N = 300
# X, y = generateBatchXor(n=N)
# X_test, y_test = generateBatchXor(n=N)

f = open("kernel_experiments.txt", "a")

f.write("-----------------Experiment-------------------\n")
spaces = "              "
## getting the alpha values of lagrange problem from svm.dual_coefs that sklearn offers:
# alpha values are by definition positive (=0 if point is not support vector, >0 if point is support vector)
# we can get their values: alphas = np.abs(svm.dual_coef_)
# since: svm.dual_coef_[i] = labels[i] * alphas[i]
# labels are either -1 or +1 and alphas are always positive.
# we get each label through: labels = np.sign(svm.dual_coef_) (kathe deigma ti label exei)
#labels[i] == -1 and alphas[i] > 0 => dual_coef_[i] < 0 and dual_coef_[i] == -alphas[i] == labels[i] * alphas[i]
# labels[i] == -1 and alphas[i] < 0 => impossible (alphas are non-negative)
# labels[i] == -1 and alphas[i]== 0 => it is not a support vector
# labels[i] == +1 and alphas[i] > 0 => dual_coef_[i] > 0 and dual_coef_[i] == alphas[i] == labels[i] * alphas[i]
# labels[i] == +1 and alphas[i] < 0 => impossible (alphas are non-negative)
# labels[i] == +1 and alphas[i]== 0 => it is not a support vector

def run_svm_for_kernel(kernel):
    clf = SVC(kernel=kernel)
    clf.fit(X, y)
    # predict on training examples - print accuracy score

    # the non-zero alphas antistoixoun sta support vectors (an ena point den einai support vector, exei a=0)
    support_vectors_number= np.abs(clf.dual_coef_).shape[1]

    sv_num_content = f'Support vectors num:{spaces}{support_vectors_number}\n'
    f.write(sv_num_content)

    def model_from_coefs_without_svm(dual_coefs ,
                                     kernel,
                                     support_vectors,
                                     support_vectors_labels,
                                     x_training):

        alphas = dual_coefs[0]
        # each support vector gives us a unique bias value
        # b = support_vectors_labels - np.sum(np.dot(alphas , kernel(support_vectors, x_training, is_x_point=False)))
        b = support_vectors_labels - np.sum(alphas * kernel(support_vectors, x_training, is_x_point=False))

        # we chose to take the mean of biases calculated
        b = np.mean(b)
        def model(x_to_predict, is_x_point=False):
            # an ta pername ola mazi:
            if is_x_point is False:
                return np.sign(np.dot(alphas, kernel(support_vectors, x_to_predict)))

            # an ta pername ena-ena ta deigmata:
            predictions = []
            for x_test in x_to_predict:
                prediction = np.sign(np.sum(np.dot(alphas, kernel(support_vectors, x_test, is_x_point=True)),axis=0) + b)
                predictions.append(prediction)

            return predictions

        return model


    support_vectors = np.take(X, clf.support_, axis=0)
    support_vectors_labels = np.take(y, clf.support_, axis=0)
    alphas = clf.dual_coef_ #me to prosimo ta theloume

    # we find the model that comes out of the coefficients for when we solved the lagrange problem
    model_from_coefs_without_svm = model_from_coefs_without_svm(alphas,
                                                                kernel,
                                                                support_vectors,
                                                                support_vectors_labels,
                                                                x_training=X)


    # for svm:
    print("--------------- For svm with kernel:\n")
    svm_train_accuracy = accuracy_score(y, clf.predict(X))
    svm_train_acc_content = f'Train accuracy score:{spaces}{spaces}{spaces}{svm_train_accuracy}\n'
    f.write(svm_train_acc_content)

    print(svm_train_acc_content)
    svm_test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    svm_test_acc_content = f'Test accuracy score:{spaces}{spaces}{spaces}{svm_test_accuracy}\n'
    f.write(svm_test_acc_content)
    print(svm_test_acc_content)

    # for model:
    print("--------------- For model with coefs:\n")
    model_train_accuracy = accuracy_score(y,  model_from_coefs_without_svm(X))
    model_train_acc_content = f'Model from dual coefficients train accuracy:{spaces}{model_train_accuracy}\n'
    f.write(model_train_acc_content)
    print(model_train_acc_content)
    model_test_acc = accuracy_score(y_test, model_from_coefs_without_svm(X_test))
    model_test_acc_content = f'Model from dual coefficients test accuracy:{spaces}{model_test_acc}\n'
    print(model_test_acc_content)
    f.write(model_test_acc_content)


seperation_line= "-----------------------\n"
# f.write(seperation_line)
# f.write("For rbf kernel:\n")
# run_svm_for_kernel('rbf')
f.write(seperation_line)
f.write("For feature map: (x1,x2,x1^2+x2^2):\n")
run_svm_for_kernel(my_kernel_1)
f.write(seperation_line)

f.write("For feature map: (x1,x2, exp(-(x1^2+x2^2))):\n")
run_svm_for_kernel(my_kernel_2)
f.write(seperation_line)
f.write("For feature map: (sqrt(2)*x1*x2, x1^2, x2^2):\n")
run_svm_for_kernel(my_kernel_3)

f.close()