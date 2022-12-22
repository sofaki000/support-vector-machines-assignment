import numpy as np

from custom_impl.my_svm import My_SVM
from data_utilities.cifar10_utilities import get_cifar, get_cifar_binary


def get_data(lower, upper, num, num_dims):
    return np.random.uniform(lower, upper, size=(num, num_dims))


def get_labels(X):
    Y = []
    for x1, x2 in X:
        if x2 < np.sin(10 * x1) / 5 + 0.3 or ((x2 - 0.8) ** 2 + (x1 - 0.5) ** 2) < 0.15 ** 2:
            Y.append(1)
        else:
            Y.append(-1)
    return np.asarray(Y)


N = 1000

data, test_data, labels, test_labels = get_cifar_binary(data_num=N)
# data = get_data(0, 1, N, 2)
# test_data = get_data(0, 1, N, 2)
# labels = get_labels(data).reshape(-1)
# test_labels = get_labels(test_data).reshape(-1)
predictions = np.ones_like(labels) * -1
# print("Max-class classifier training set accuracy: ", np.mean(np.equal(predictions, labels)) * 100, "%")
model = My_SVM()
model.fit_data(data, labels)

correct = 0
samples_size = test_data.shape[0]

for i in range(samples_size):
    x = test_data[i]
    y = test_labels[i]
    prediction = model.predict(x)

    if prediction==y:
        correct+=1

accuracy = (correct/N) * 100

f = open("from_scratch_experiments.txt", "a")
f.write(f"No b , samples={N}\n")
accuracy_content = f'Accuracy is {accuracy}%'
f.write(f'{accuracy_content}\n')
print(accuracy_content)
f.close()