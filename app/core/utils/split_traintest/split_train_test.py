import pandas as pd
from sklearn.model_selection import train_test_split


def train_tes_split(df, SEED, test_size=0.2):  # 分训练集和测试集

    X = df[df.columns[1:]]
    y = df[df.columns[0]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED)
    df_train_X = pd.DataFrame(X_train)
    df_train_y = pd.Series(y_train)
    df_train = pd.concat([df_train_y, df_train_X], axis=1)

    df_test_x = pd.DataFrame(X_test)
    df_test_y = pd.Series(y_test)
    df_test = pd.concat([df_test_y, df_test_x], axis=1)

    return df_train, df_test
