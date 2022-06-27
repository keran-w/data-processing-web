import pandas as pd


def vars_types(df):
    indep = df.columns[1:]
    quan = []
    mult = []
    binary = []  # 二值变量
    mult_order = []
    mult_disorder = []
    var_len = {}

    for i in indep:
        f = len(df[i].dropna().unique())
        if f > 10:
            quan.append(i)
        elif 2 < f <= 10:
            mult.append(i)
        elif f == 2:
            binary.append(i)
        var_len[i] = f

    print('var_len:', var_len)

    mult_order1 = input('请输入多值有序的变量：').split(',')
    mult_disorder1 = input('请输入多值无序的变量：').split(',')
    print('连续变量有：', quan)
    qua = input('请删除连续变量：').split(',')
    qua1 = input('请补充连续变量：').split(',')

    if mult_order1 != ['']:
        for l in mult_order1:
            mult.remove(l)
            mult_order.append(l)
    else:
        print('无多值有序变量')

    if mult_disorder1 != ['']:
        for l in mult_disorder1:
            mult.remove(l)
            mult_disorder.append(l)
    else:
        print('无多值无序变量')

    if qua != ['']:
        for i in qua:
            quan.remove(i)

    if qua1 != ['']:
        for l in qua1:
            binary.remove(l)
            quan.append(l)
    return mult_order, binary, quan, mult_disorder


def var_type(df):
    dep = df.columns[0]
    yvar_type = []
    if df.iloc[:, 0].dtype == float:
        yvar_type.append('quan')
    else:
        f = len(df[dep].unique())
        if f > 2:
            var_type = input('请输入因变量类型,T1-多值有序，T2-多值无序，T3-连续变量:')
            if 'T2' in var_type:
                yvar_type.append('mult_disorder')
            elif 'T3' in var_type:
                yvar_type.append('quan')
            elif 'T1' in var_type:
                yvar_type.append('mult_order')
        elif f == 2:
            yvar_type.append('binary')

    return yvar_type


def var_ty(df, yvar_type, mult_order, binary, quan, mult_disorder, RESULT_PATH):
    res = []
    if 'binary' in yvar_type:  # 因变量为二值变量
        if len(quan) > 0:
            df_quant_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_quant_quan.to_excel(
                RESULT_PATH + 'tmp/' + "df_quant_quan_binary.xlsx", index=False)
            res.append(("df_quant_quan_binary.xlsx", 'binary'))
        if len(mult_order) > 0:
            df_chi_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_chi_order.to_excel(RESULT_PATH + 'tmp/' +
                                  "df_chisquare_order_binary.xlsx", index=False)
            res.append(("df_chisquare_order_binary.xlsx", 'binary'))
        if len(mult_disorder) > 0:
            df_chi_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_chi_disorder.to_excel(RESULT_PATH + 'tmp/' +
                                     "df_chisquare_disorder_binary.xlsx", index=False)
            res.append(("df_chisquare_disorder_binary.xlsx", 'binary'))
        if len(binary) > 0:
            df_chi_binary = pd.concat([df.iloc[:, 0], df[binary]], axis=1)
            df_chi_binary.to_excel(RESULT_PATH + 'tmp/' +
                                   "df_chisquare_binary_binary.xlsx", index=False)
            res.append(("df_chisquare_binary_binary.xlsx", 'binary'))
    elif 'mult_disorder' in yvar_type:  # 因变量为多值无序
        if len(quan) > 0:
            df_quant_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_quant_quan.to_excel(RESULT_PATH + 'tmp/' +
                                   "df_quant_quan_mult_disorder.xlsx", index=False)
            res.append(("df_quant_quan_mult_disorder.xlsx", 'mult_disorder'))
        if len(mult_order) > 0:
            df_chi_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_chi_order.to_excel(RESULT_PATH + 'tmp/' +
                                  "df_chisquare_order_mult_disorder.xlsx", index=False)
            res.append(
                ("df_chisquare_order_mult_disorder.xlsx", 'mult_disorder'))
        if len(mult_disorder) > 0:
            df_chi_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_chi_disorder.to_excel(RESULT_PATH + 'tmp/' +
                                     "df_chisquare_disorder_mult_disorder.xlsx", index=False)
            res.append(
                ("df_chisquare_disorder_mult_disorder.xlsx", 'mult_disorder'))
        if len(binary) > 0:
            df_chi_binary = pd.concat([df.iloc[:, 0], df[binary]], axis=1)
            df_chi_binary.to_excel(RESULT_PATH + 'tmp/' +
                                   "df_chisquare_binary_mult_disorder.xlsx", index=False)
            res.append(
                ("df_chisquare_binary_mult_disorder.xlsx", 'mult_disorder'))
    elif 'mult_order' in yvar_type:  # 因变量为多值有序
        if len(quan) > 0:
            df_spe_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            # stats.spearmanr()
            df_spe_quan.to_excel(RESULT_PATH + 'tmp/' + "df_spe_quan.xlsx", index=False)
            res.append(("df_spe_quan.xlsx", 'mult_order'))
        if len(mult_order) > 0:
            df_spe_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_spe_order.to_excel(RESULT_PATH + 'tmp/' + "df_spe_order.xlsx", index=False)
            res.append(("df_spe_order.xlsx", 'mult_order'))
        if len(mult_disorder) > 0:
            df_nonparametric_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            # qualitative.nonparametric_test()
            df_nonparametric_disorder.to_excel(RESULT_PATH + 'tmp/' +
                                               "df_nonparametric_disorder.xlsx", index=False)
            res.append(("df_nonparametric_disorder.xlsx", 'mult_order'))
        if len(binary) > 0:
            df_nonparametric_binary = pd.concat(
                [df.iloc[:, 0], df[binary]], axis=1)
            df_nonparametric_binary.to_excel(RESULT_PATH + 'tmp/' +
                                             "df_nonparametric_binary.xlsx", index=False)
            res.append(("df_nonparametric_binary.xlsx", 'mult_order'))
    elif 'quan' in yvar_type:  # 因变量为连续变量
        if len(quan) > 0:
            df_cor_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_cor_quan.to_excel(RESULT_PATH + 'tmp/' + "df_cor_quan.xlsx", index=False)
            res.append(("df_cor_quan.xlsx", 'quan'))
        if len(mult_order) > 0:
            df_quant_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_quant_order.to_excel(
                RESULT_PATH + 'tmp/' + "df_cor_order.xlsx", index=False)
            res.append(("df_cor_order.xlsx", 'quan'))
        if len(mult_disorder) > 0:
            df_quant_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_quant_disorder.to_excel(
                RESULT_PATH + 'tmp/' + "df_quant_disorder.xlsx", index=False)
            res.append(("df_quant_disorder.xlsx", 'quan'))
        if len(binary) > 0:
            df_quant_binary = pd.concat([df.iloc[:, 0], df[binary]], axis=1)
            df_quant_binary.to_excel(
                RESULT_PATH + 'tmp/' + "df_quant_binary.xlsx", index=False)
            res.append(("df_quant_binary.xlsx", 'quan'))
    return res
