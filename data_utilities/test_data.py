from sklearn.datasets import make_blobs


def get_test_data(num_samples=1000, train_percentage=0.8):

    indx_split = int(num_samples * train_percentage)
    dataX, datay = make_blobs(n_samples=num_samples, centers=2, n_features=2, cluster_std=2, random_state=2)
    X, x_test = dataX[:indx_split], dataX[indx_split:]
    y, y_test = datay[:indx_split], datay[indx_split:]
    return X, x_test, y,y_test