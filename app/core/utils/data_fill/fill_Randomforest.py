import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def random_forest(df, var_type_dict):
    dict = var_type_dict
    quan = dict['quan']

    null_columns_num = df.isnull().sum(
        axis=0)[df.isnull().sum(axis=0) > 0]  # 得到缺失值的列名
    df_null_value = pd.DataFrame(null_columns_num.index, columns=['feature'])
    df_null_value['num of missing'] = null_columns_num.values
    df_null_value['imputing value'] = np.nan

    df_new = df.copy()
    for miss_column in null_columns_num.sort_values().index:
        # 删除含有空值的列.
        df_nonempty = df_new.dropna(axis=1, how='any', inplace=False)
        fill_column = df_new.loc[:, miss_column]  # 含有空值的列

        ytrain = fill_column[fill_column.notnull()]  # 被选中填充的特征矩阵T中的非空值
        ytest = fill_column[fill_column.isnull()]  # 被选中填充的特征矩阵T中的空值
        # 新特征矩阵上，被选出来要填充的特征的非空值对应的样本
        Xtrain = df_nonempty.iloc[ytrain.index, :]
        Xtest = df_nonempty.iloc[ytest.index, :]  # 空值对应的样本

        if miss_column in quan:
            rf = RandomForestRegressor(random_state=1, n_jobs=-1)
        else:
            rf = RandomForestClassifier(random_state=1, n_jobs=-1)

        rf = rf.fit(Xtrain, ytrain)
        filled_value = rf.predict(Xtest)  # predict接口预测得到的结果就是用来填充空值的那些值

        for i, row_idx in enumerate(Xtest.index):
            df_new.loc[row_idx:row_idx, miss_column].fillna(
                filled_value[i], inplace=True)

        df_null_value['imputing value'] = df_null_value['imputing value'].mask(df_null_value['feature'] == miss_column,
                                                                               str(filled_value))
    return df_new
