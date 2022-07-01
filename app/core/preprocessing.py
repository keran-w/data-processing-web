from .. import settings

SAMPLING_METHODS = settings.SAMPLING_METHODS[1].keys()
IMPUTE_METHODS = settings.IMPUTE_METHODS[1].keys()
SELE_METHODS = settings.SELE_METHODS[1].keys()


def seed_everything(seed=20):
    '''set seed for all'''
    import os
    import random
    import numpy as np
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def data_cleaning(data, RESULT_PATH, debug=True):
    '''
        数据清洗：
            1. 删除标签有缺失的行
            2. 删除缺失值大于80%的列
            3. 删除单个类别大于80%的列
            4. 删除变异系数小于0.1的列
    '''

    from .utils.data_preprocessing.data_prepro import del_label_line, del_miss_columns, del_category_colunms, cv_delete

    # 1. 删除标签有缺失的行
    data = del_label_line(data)

    # 2. 删除缺失值大于80%的列
    data = del_miss_columns(data)

    # 3. 删除单个类别大于80%的列
    data = del_category_colunms(data)

    # 4. 删除变异系数小于0.1的列
    data = cv_delete(data)

    # 保存数据清洗后的数据
    if debug:
        data.to_csv(RESULT_PATH + 'logging/step_1_data_cleaning.csv',
                    encoding='utf-8-sig', index=None)
    return data


def mult_disorder_to_dummies(data, var_type_dict, RESULT_PATH):
    '''
        多值无序变量设置为哑变量
    '''
    from .utils.data_preprocessing.data_prepro import get_dummis
    data = get_dummis(data, var_type_dict, RESULT_PATH)

    return data


def data_filling(data, var_type_dict, RESULT_PATH, IMPUTE_METHODS, imp_method_=None):
    '''
        6种数据填充方式: 
            Simple
            KNN
            ISVD
            Imput
            rf
            optimal
    '''
    from .utils.data_fill.data_fil import data_impute

    data_filling_dict = {}
    for method in IMPUTE_METHODS:
        if imp_method_ is not None and method not in imp_method_:
            continue

        print(f'Impute method: {method}')
        data_imp = data_impute(data, method, var_type_dict)
        data_imp.to_excel(RESULT_PATH + f'tmp/imp_{method}.xlsx', index=False)
        data_filling_dict[method] = data_imp
    return data_filling_dict


def train_test_split(data, SEED):
    '''
        将原数据分为80%测试集和20%测试集
    '''
    from .utils.split_traintest.split_train_test import train_tes_split
    df_train, df_test = train_tes_split(data, SEED)
    return df_train, df_test


def data_sampling(data, SEED, RESULT_PATH, sampling_method_=None, title=''):
    '''
        6种数据采样:
        SMO
        SSMO
        BSMO
        ADA
        ROS
        SMN
    '''
    from .utils.data_sampling import data_samp
    data_sampling_dict = {}

    for sampling_method in SAMPLING_METHODS:
        if sampling_method_ is not None and sampling_method not in sampling_method_:
            continue

        print(f'{title} Sample Method: {sampling_method}')
        data_smpl = data_samp.data_sampling(data, sampling_method, SEED)
        data_smpl.to_excel(
            RESULT_PATH + f'tmp/data_smpl_{sampling_method}.xlsx', index=False)
        data_sampling_dict[sampling_method] = data_smpl

    return data_sampling_dict


def feature_selection(data, SEED, SELE_METHODS, sele_method_=None):
    '''
        # 特征筛选12种
    '''
    from .utils.feature_selection import feature_sele

    variable_selected_dict = {}

    for sele_method in SELE_METHODS:
        try:
            if sele_method_ is not None and sele_method not in sele_method_:
                continue
            print(f'Feature Selection Method: {sele_method}')
            variable_selected_dict[sele_method] = feature_sele.feature_selection(
                data, sele_method, SEED)
        except Exception as e:
            print(f'Feature Selection Error: {e}')

    return variable_selected_dict


def process_data(CFG, RESULT_PATH):
    
    print()    
    print()    
    print(CFG)
    print()    
    print()    
    data = CFG['data']
    tgt_col = CFG['tgt_col']
    imp_method_ = CFG['imp_method']
    sampling_method_ = CFG['sampling_method']
    sele_method_ = CFG['sele_method']
    variables =  CFG['variables']
    var_types = CFG['var_types']
    var_type_dict = {k:[] for k in ['binary', 'quan', 'mult_order', 'mult_disorder']}
    for k, v in zip(variables, var_types):
        var_type_dict[v].append(k)
    
    SEED = CFG['seed']

    data = data[[tgt_col] + [col for col in data.columns if col != tgt_col]]

    # 数据清洗
    data = data_cleaning(data, RESULT_PATH)

    # 多值无序变量设置为哑变量
    data = mult_disorder_to_dummies(data, var_type_dict, RESULT_PATH)

    # 数据填充6种
    if imp_method_[0] == '':
        data_filling_dict = {'': data}
    else:
        data_filling_dict = data_filling(
            data, var_type_dict, RESULT_PATH, IMPUTE_METHODS, imp_method_)

    all_keys = []

    for imp_method, data_imp in data_filling_dict.items():
        data_imp = data_filling_dict[imp_method]

        # 分为80%测试集和20%测试集
        data_imp_train, data_imp_test = train_test_split(data_imp, SEED)

        # 数据采样6种
        if sampling_method_[0] == '':
            data_train_dict = {'': data_imp_train}
            data_test_dict = {'': data_imp_test}
        else:
            data_train_dict = data_sampling(
                data_imp_train, SEED, RESULT_PATH, sampling_method_, 'Train Data')
            data_test_dict = data_sampling(
                data_imp_test, SEED, RESULT_PATH, sampling_method_, 'Test Data')

        for sample_method in data_train_dict.keys():
            data_train, data_test = data_train_dict[sample_method], data_test_dict[sample_method]

            # 特征筛选 12种
            if sele_method_[0] == '':
                variable_selected_dict = {'':list(data_train.columns[1:])}
            else:    
                variable_selected_dict = feature_selection(
                    data_train, SEED, sele_method_)

            for sele_method in variable_selected_dict.keys():
                vars_sele = [tgt_col] + variable_selected_dict[sele_method]
                data_train_sele = data_train[vars_sele]
                data_test_sele = data_test[vars_sele]

                key = f'{imp_method}_{sample_method}_{sele_method}'
                all_keys.append(key)

                data_train_sele.to_csv(
                    RESULT_PATH + f'data/{key}_train.csv', encoding='utf-8-sig', index=False)
                data_test_sele.to_csv(
                    RESULT_PATH + f'data/{key}_test.csv', encoding='utf-8-sig', index=False)
    return var_type_dict
