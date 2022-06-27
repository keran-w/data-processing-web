from sklearn import preprocessing


def get_dataset(path1, path2, isMaxMin=True):
    """输入训练集和测试集"""
    df_train = path1
    df_test = path2
    x_train = df_train.iloc[:, 1:]
    y_train = df_train.iloc[:, 0]
    x_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0]

    if isMaxMin:
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    else:
        minMaxScaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = minMaxScaler.transform(x_train)
        x_test = minMaxScaler.transform(x_test)

    return x_train, x_test, y_train, y_test
