from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data['data'], data['target']

x_train, x_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.svm import SVC
# gramiko svm
from sklearn.svm import LinearSVC

#
# for C in [0.01, 0.1, 1 ,10,100,1000, 10000]:
#     svm = LinearSVC(loss='hinge', multi_class='ovr', C=C, max_iter=100000)  # one versus rest, another: one versus one
#     svm.fit(x_train, y_train)
#
#     print(f'C:{C}, Train : {100 * svm.score(x_train, y_train)}, test:{100 * svm.score(x_test, y_test)}')
#
#     print(svm.support_vectors_.shape)
#
#
# #micro C -> beltistopoioume to margin ginetai pio megalo
# # ta perissotera datasets einai sto line, mono 4 deigmata einai ektos margin-> xalia accuracy
# # kai polla support vectors
# svm = SVC(C=1, kernel='linear', max_iter=100000)
# svm.fit(x_train, y_train)
# print(f'Train : {100 * svm.score(x_train, y_train)}, test:{100 * svm.score(x_test, y_test)}')
#
# # tha doume support vectors
# print(svm.support_vectors_.shape)

# polyonimikos kernel
# for C in [0.01, 0.1, 1 ,10,100,1000, 10000]:
#     svm_poly = SVC(C=1, kernel='poly', degree=3, max_iter=100000)
#     svm_poly.fit(x_train, y_train)
#
#     print(f'C:{C}, Train : {100 * svm_poly.score(x_train, y_train)}, test:{100 * svm_poly.score(x_test, y_test)}')
#
#     print(svm_poly.support_vectors_.shape)

for gamma in [0.01, 0.1, 1 ,10,100,1000, 10000]:
    svm_poly = SVC(C=1, kernel='rbf', gamma=gamma,decision_function_shape='ovr', max_iter=100000)
    svm_poly.fit(x_train, y_train)

    print(f'gamma:{gamma}, Train : {100 * svm_poly.score(x_train, y_train)}, test:{100 * svm_poly.score(x_test, y_test)}')

    print(svm_poly.support_vectors_.shape)