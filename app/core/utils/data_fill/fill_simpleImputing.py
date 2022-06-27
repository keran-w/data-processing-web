import pandas as pd
import numpy as np


def simple_imputing(df_org, var_type_dict):
    mult_disorder = var_type_dict['mult_disorder']
    quan = var_type_dict['quan']
    binary = var_type_dict['binary']
    mult_order = var_type_dict['mult_order']

    null_columns_num = df_org.isnull().sum(
        axis=0)[df_org.isnull().sum(axis=0) > 0]  # 缺失值的列名和缺失数量
    df_null_value = pd.DataFrame(null_columns_num.index, columns=[
                                 'feature'])  # 形成dataFrame，缺失值的列名为feature

    df_null_value['num of missing'] = null_columns_num.values  # 缺失值的数量
    # The empty cell are used to save the filled values.
    df_null_value['imputing value'] = np.nan

    df_new = df_org.copy()
    for column_name in df_null_value['feature'].values.tolist():
        data = df_org[column_name]  # 列举出每个变量的缺失值
        filled_value = 0
        if column_name in quan:  # 连续变量,使用均值
            filled_value = data.mean()
        elif column_name in mult_order:  # 多值有序,使用中位数
            filled_value = data.median()
        elif column_name in mult_disorder:  # 多值无序，使用众数
            filled_value = data.mode()[0]  # 多值无序
        elif column_name in binary:
            filled_value = data.mode()[0]  # 二值变量，使用众数

        df_new[column_name].fillna(filled_value, inplace=True)

        df_null_value['imputing value'] = df_null_value['imputing value'].mask(
            df_null_value['feature'] == column_name, filled_value)

    return df_new
