from scipy import stats
from utils.univariate_analysis.tool import format_float


def judge_normal(data: list):  # 判断正态分布
    judge_list = []
    for _data in data:
        _, p_sha = stats.shapiro(_data)
        judge = 'NO' if p_sha < 0.05 else 'Yes'
        judge_list.append(judge)

    if 'NO' in judge_list:
        res = False
    else:
        res = True
    return res


def nonparametric_test(data: list):  # 非参数检验
    group_num = len(data)
    assert group_num >= 2
    if group_num == 2:
        _, P = stats.mannwhitneyu(data[0], data[1], alternative='two-sided')
        method = 'mannwhitneyu'
        return format_float(_, 4), format_float(P, 4), method
    else:
        _, P = stats.kruskal(*data)
        method = 'Kruskal-Wallis'
        return format_float(_, 4), format_float(P, 4), method


def parametric_test(data: list):  # 参数检验
    group_num = len(data)
    assert group_num >= 2
    _, p_levene = stats.levene(*data)  # 方差分析
    equal_var = True if p_levene > 0.05 else False
    if group_num == 2:
        if equal_var == True:
            _, p = stats.ttest_ind(data[0], data[1], equal_var=True)
            method = 'independent_test'
            return format_float(_, 4), format_float(p, 4), method
        else:
            _, p = stats.ttest_ind(data[0], data[1], equal_var=False)
            method = 'calibration_test'
            return format_float(_, 4), format_float(p, 4), method
    else:
        if equal_var == True:
            _, P = stats.f_oneway(*data)
            method = 'f_oneway'
            return format_float(_, 4), format_float(P, 4), method
        else:
            P = nonparametric_test(data)
            return P


def univariate_quantitative(data: list):
    group_num = len(data)
    assert group_num >= 2
    res = judge_normal(data)
    if res == True:
        P = parametric_test(data)
    else:
        P = nonparametric_test(data)
    return P
