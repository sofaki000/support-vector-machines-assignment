from sklearn.preprocessing import MinMaxScaler


def scale_data(X, y):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    return X, y