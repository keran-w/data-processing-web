from scipy import stats
import pandas as pd
import numpy as np
import scipy as sp
from .tool import format_float


def two_disorderly(filepath):  # 双向无序
    result_columns = ['variable', 'group', 'frequency_0',
                      'frequency_1', 'statistics', 'P values', 'method']
    df_result = pd.DataFrame(
        data=[[np.nan, np.nan, 'Y=0', 'Y=1', np.nan, np.nan, np.nan]], columns=result_columns)

    df = pd.read_excel(filepath)

    for i in df.columns[1:]:
        cro_data = pd.crosstab(df.iloc[:, 0], df[i])
        cro_data_t = cro_data.T

        f = len(df[i].dropna().unique())

        if f > 2:
            _, p, x2, x3 = stats.chi2_contingency(cro_data)
            # _, P = stats.chisquare(cro_data, axis=None)
            method = 'chisquare'
            paramaters = (i, format_float(_, 4), format_float(p, 4), method)

        # 如果变量的长度大于2（有2个以上的变量），则使用卡方检验
        else:
            x2, p, f, l = sp.stats.chi2_contingency(cro_data, correction=False)
            m = np.min(l)
            """
            l:理论频数
            m:最小的理论频数
            """
            if m >= 5:
                _, P, x2, x3 = stats.chi2_contingency(cro_data)
                # _, P = stats.chisquare(cro_data, axis=None)
                method = 'chisquare'
                """
                如果最小理论频数大于等于5，则使用卡方检验
                 """
            else:
                _, P = stats.fisher_exact(cro_data, alternative='two-sided')
                method = 'fisher'
                """
                如果最小理论频数小于5，则使用fisher检验
                 """

            paramaters = (i, format_float(_, 4), format_float(P, 4), method)
            """
            收集变量i，_和P值
            """

        # 构建需要的格式
        df_cro_data_t = pd.DataFrame(
            np.concatenate([np.array([[np.nan] for _ in range(cro_data_t.shape[0])]),  # shape[0]输出矩阵的行数
                            # reshape(-1,1)转换成1列
                            cro_data_t.index.values.reshape(-1, 1),
                            cro_data_t.values,
                            np.array([[np.nan]
                                     for _ in range(cro_data_t.shape[0])]),
                            np.array([[np.nan]
                                     for _ in range(cro_data_t.shape[0])]),
                            np.array([[np.nan]
                                     for _ in range(cro_data_t.shape[0])]),
                            ], axis=1), columns=result_columns)  # axis=1表示对应行的数组进行拼接

        _paramaters = pd.DataFrame(
            [[paramaters[0], np.nan, np.nan, np.nan, *paramaters[1:]]], columns=result_columns)

        df_result = pd.concat([df_result, _paramaters, df_cro_data_t], axis=0)
    return df_result

# two_disorderly('tape.xlsx')
