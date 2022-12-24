import numpy as np
from custom_impl.my_svm import My_SVM
from data_utilities.cifar10_utilities import get_cifar_binary
import time
from utilities.metric_utilities import get_metrics
from utilities.pca_utilities import get_pca_data

start_time = time.time()

experiments_file_name = "pca_experiments.txt"

N = 2000

data, test_data, labels, test_labels = get_cifar_binary(data_num=N)

data, test_data = get_pca_data(data, test_data)

model = My_SVM(kernel_type='rbf')
sv_num = model.fit_data(data, labels)
train_duration =  time.time() - start_time

correct = 0
samples_size = test_data.shape[0]

model_predictions = []

for i in range(samples_size):
    print(f"Testing num {i}...")
    x = test_data[i]
    y = test_labels[i]
    prediction = model.predict(x)
    model_predictions.append(prediction)
    if prediction==y:
        correct+=1


actual, predicted = test_labels, np.array(model_predictions)
metrics_content = get_metrics('confusion_matrix_rbf_kernel', actual, predicted)

accuracy = (correct/N) * 100


f = open(experiments_file_name, "a")
f.write(f"pca rbf , samples={N}\n")

time_content = f'Training time:{train_duration:.2f}s'
print(time_content)
f.write(f'{time_content}\n')

number_of_a_class_predictions = np.count_nonzero(np.array(model_predictions)==1)
number_of_b_class_predictions = np.count_nonzero(np.array(model_predictions)==-1)
predictions_content = f'Predicted A class:{number_of_a_class_predictions}.\nPredicted b class:{number_of_b_class_predictions}'
print(predictions_content)
f.write(f'{predictions_content}\n')


num_data, num_features = data.shape
samples_num_content = f'Num of samples:{num_data}'
sv_num_content = f'Num of support vectors {sv_num}'
print(samples_num_content)
print(sv_num_content)
f.write(f'{samples_num_content}\n')
f.write(f'{sv_num_content}\n')


accuracy_content = f'Accuracy is {accuracy}%'
f.write(f'{accuracy_content}\n')

f.write(f'{metrics_content}\n')

print(accuracy_content)
f.close()