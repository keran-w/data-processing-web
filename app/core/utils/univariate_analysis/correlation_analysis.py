from scipy import stats
import pandas as pd
import numpy as np
from utils.univariate_analysis.tool import format_float


def correlation_analysis(filepath):  # 相关性分析
    df = pd.read_excel(filepath)
    para = []
    columns_name = ['variable', 'Mean±SD', 'statistic', 'pvalue', 'method']

    for l in df.columns[1:]:
        for j in df.columns[0]:
            df_ad = df.loc[:, [j, l]]
            df_ad = df_ad.dropna()
            f = df[j].dropna()
            _, p_sha = stats.shapiro(f)

            m = np.mean(df[l]).__float__()
            n = np.std(df[l]).__float__()
            t = f"{format_float(m, 4)} ± {format_float(n, 4)}"

            if p_sha < 0.05:
                _, P = stats.spearmanr(df_ad)
                method = 'spearmanr'

                paramaters1 = (l, t, format_float(_, 4),
                               format_float(P, 4), method)
                para.append(paramaters1)

            else:
                _, p_sha = stats.shapiro(df_ad[l])

                if p_sha < 0.05:
                    _, P = stats.spearmanr(df_ad)
                    method = 'spearmanr'
                else:
                    _, P = stats.pearsonr(
                        np.array(df_ad.iloc[:, 0]), np.array(df_ad.iloc[:, 1]))
                    method = 'pearsonr'

                paramaters2 = (l, t, format_float(_, 4),
                               format_float(P, 4), method)
                para.append(paramaters2)

    df_var = pd.DataFrame(para, columns=columns_name)
    df_var.set_index('variable', inplace=True)
    return df_var
    # return df_var.to_excel('correlation_analysis_F1.xlsx')

# correlation_analysis('F1-cor.xlsx')
