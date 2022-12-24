
from data_utilities.cifar10_utilities import get_cifar_binary


X_train, test_data, y_train, test_labels = \
    get_cifar_binary(data_num=20,
                     percentage_of_data_to_keep=0.5)
