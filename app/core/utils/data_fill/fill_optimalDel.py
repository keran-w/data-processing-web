import pandas as pd
import numpy as np


def fill_nan(x):  # 填充值
    x[pd.Series.isna(x) == False] = 100
    null_n = pd.Series.isnull(x).sum()
    pd.Series.fillna(x, value=round((len(x)-null_n) / len(x) * 100), inplace=True)
    return x

def optimal_del_func(df):
    df_for_del = df.copy(deep=True)
    df_del_metrix = df_for_del.apply(lambda x: fill_nan(x), axis=1).astype(np.int32)
    sum_columns = df_del_metrix.sum(axis=0)
    del_sort = sum_columns.sort_values().index.tolist()
    del_num_sample_num = {}
    del_num_columns = {}
    opti_index_cells_dict = {}
    for del_num in range(len(df.columns)+1):
        df_1 = df.copy(deep=True)
        df_1 = df_1.drop(columns=del_sort[:del_num])
        df_1 = df_1.dropna(axis=0, how='any')
        sample_num = df_1.shape[0]
        left_columns = df_1.columns.tolist()
        left_col_num = len(left_columns)
        opti_index_cells = sample_num*left_col_num
        opti_index_cells_dict[del_num]=opti_index_cells
        del_num_sample_num[del_num]=sample_num
        del_num_columns[del_num]=left_columns
    return opti_index_cells_dict,del_num_sample_num,del_num_columns

def find_opt_del_mode(df):
    opti_index_cells_dict, del_num_sample_num, del_num_columns=optimal_del_func(df)
    opt_num = max(opti_index_cells_dict.values())

    for key, value in opti_index_cells_dict.items():
        if value == opt_num:
            opti_del_col_num=key
    opti_sample_num = del_num_sample_num[opti_del_col_num]
    opti_left_col = del_num_columns[opti_del_col_num]
    df_out = df[opti_left_col]
    df_out = df_out.dropna(axis=0, how='any')

    # return opti_del_col_num,opti_sample_num,opti_left_col,del_num_sample_num,opti_index_cells_dict,df_out
    return df_out
