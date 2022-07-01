import pandas as pd


def var_type(vars):
    num_unique = len(set(vars))
    if num_unique > 10:
        return 'quan'
    elif 2 < num_unique <= 10:
        return 'mult_disorder'
    elif num_unique == 2:
        return 'binary'


def all_var_type(data_df):
    return {var: var_type(var) for var in data_df.columns}
