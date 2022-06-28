import pandas as pd
import numpy as np
from fancyimpute import KNN, IterativeSVD, IterativeImputer
from . import fill_simpleImputing
from . import fill_Randomforest
from . import fill_optimalDel


def data_impute(df, impute_method, var_type_dict):  # 数据填充
    df_impute = np.array(df)

    if impute_method == 'Simple':
        df_imp = fill_simpleImputing.simple_imputing(df, var_type_dict)
    elif impute_method == 'KNN':
        df_np = KNN(k=3).fit_transform(df_impute)
        df_imp = pd.DataFrame(df_np, columns=df.columns)
    elif impute_method == 'ISVD':
        df_np = IterativeSVD(verbose=False).fit_transform(df_impute)
        df_imp = pd.DataFrame(df_np, columns=df.columns)
    elif impute_method == 'Imput':
        df_np = IterativeImputer(verbose=False).fit_transform(df_impute)
        df_imp = pd.DataFrame(df_np, columns=df.columns)
    elif impute_method == 'rf':
        df_imp = fill_Randomforest.random_forest(df, var_type_dict)
    elif impute_method == 'optimal':
        df_imp = fill_optimalDel.find_opt_del_mode(df)

    return df_imp
