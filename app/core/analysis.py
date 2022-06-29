import pandas as pd


def var_ty(df, yvar_type, var_type_dict, RESULT_PATH):
    mult_order = var_type_dict['mult_order']
    binary = var_type_dict['binary']
    quan = var_type_dict['quan']
    mult_disorder = var_type_dict['mult_disorder']

    res = []
    if yvar_type == 'binary':
        if len(quan) > 0:
            df_quant_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_quant_quan.to_excel(RESULT_PATH + 'tmp/' +
                                   "df_quant_quan_binary.xlsx", index=False)
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
    elif yvar_type == 'mult_disorder':
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
    elif yvar_type == 'mult_order':
        if len(quan) > 0:
            df_spe_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)

            df_spe_quan.to_excel(RESULT_PATH + 'tmp/' +
                                 "df_spe_quan.xlsx", index=False)
            res.append(("df_spe_quan.xlsx", 'mult_order'))

        if len(mult_order) > 0:
            df_spe_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_spe_order.to_excel(RESULT_PATH + 'tmp/' +
                                  "df_spe_order.xlsx", index=False)
            res.append(("df_spe_order.xlsx", 'mult_order'))

        if len(mult_disorder) > 0:
            df_nonparametric_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_nonparametric_disorder.to_excel(RESULT_PATH + 'tmp/' +
                                               "df_nonparametric_disorder.xlsx", index=False)
            res.append(("df_nonparametric_disorder.xlsx", 'mult_order'))

        if len(binary) > 0:
            df_nonparametric_binary = pd.concat(
                [df.iloc[:, 0], df[binary]], axis=1)
            df_nonparametric_binary.to_excel(RESULT_PATH + 'tmp/' +
                                             "df_nonparametric_binary.xlsx", index=False)
            res.append(("df_nonparametric_binary.xlsx", 'mult_order'))
    elif yvar_type == 'quan':
        if len(quan) > 0:
            df_cor_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_cor_quan.to_excel(RESULT_PATH + 'tmp/' +
                                 "df_cor_quan.xlsx", index=False)
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
