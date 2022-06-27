import pandas as pd
import numpy as np
from utils.univariate_analysis import qualitaive_analysis
from utils.univariate_analysis.tool import format_float


# filepath = 'test.xlsx'#因变量为二值/多值，自变量为连续变量
# df = pd.read_excel(filepath)
columns_name = ['variable', 'group', 'Mean±SD',
                'statistics', 'P values', 'method']


def univariate_quantitative_methods(df):
    result = {}
    all_df_vars = pd.DataFrame([], columns=columns_name)

    for i in df.columns[1:]:
        # for j in df.columns[0]:
        j = df.columns[0]
        df_ad1 = df.loc[:, [j, i]]
        df_ad = df_ad1.dropna()

        data_list = []
        df_mean_sd_all = pd.DataFrame([], columns=columns_name)
        for name, group in df_ad.groupby(j):
            data_list.append(group[i].values)
            m = np.mean(group[i].values).__float__()  # 计算均值
            n = np.std(group[i].values).__float__()  # 计算标准差
            t = f"{format_float(m, 4)} ± {format_float(n, 4)}"

            df_mean_sd = pd.DataFrame(
                [[np.nan, name, t, np.nan, np.nan, np.nan]], columns=columns_name)
            df_mean_sd_all = df_mean_sd_all.append(df_mean_sd)

        s_, p_, m_ = qualitaive_analysis.univariate_quantitative(data_list)
        df_var = pd.DataFrame(
            [[i, np.nan, np.nan, s_, p_, m_]], columns=columns_name)
        df_vars = df_var.append(df_mean_sd_all)

        all_df_vars = all_df_vars.append(df_vars)

        result[j] = all_df_vars
    return result

# res=univariate_quantitative_methods(df)


# for var_name,stat_data in res.items():
#     stat_data.to_excel(f'stat_{var_name}_p.xlsx',encoding='utf-8',index=False)


# filepath = 'F11.xlsx'#因变量为连续变量，自变量为二值/多值
# df = pd.read_excel(filepath)
# columns_name = ['variable','group','Mean±SD','statistics','P values','method']
def univariate_quantitative_methods1(df):  # 因变量为连续变量，自变量为多值，二值
    result = {}
    all_df_vars = pd.DataFrame([], columns=columns_name)
    for i in df.columns[1:]:
        for j in df.columns[0]:

            df_ad1 = df.loc[:, [j, i]]
            df_ad = df_ad1.dropna()

            data_list = []
            df_mean_sd_all = pd.DataFrame([], columns=columns_name)
            for name, group in df_ad.groupby(i):
                data_list.append(group[j].values)
                m = np.mean(group[j].values).__float__()
                n = np.std(group[j].values).__float__()
                t = f"{format_float(m, 4)} ± {format_float(n, 4)}"
                df_mean_sd = pd.DataFrame(
                    [[np.nan, name, t, np.nan, np.nan, np.nan]], columns=columns_name)
                df_mean_sd_all = df_mean_sd_all.append(df_mean_sd)

            s_, p_, m_ = qualitaive_analysis.univariate_quantitative(data_list)
            df_var = pd.DataFrame(
                [[i, np.nan, np.nan, s_, p_, m_]], columns=columns_name)
            df_vars = df_var.append(df_mean_sd_all)

            all_df_vars = all_df_vars.append(df_vars)

            result[j] = all_df_vars
    return result
#
# res=univariate_quantitative_methods1(df)
#
# for var_name,stat_data in res.items():
#     stat_data.to_excel(f'stat_F1.xlsx',encoding='utf-8',index=False)
