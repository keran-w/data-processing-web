import os
import sys
import json
import shutil
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import preprocessing

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import *
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

import yaml
import shap

import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'


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

    from utils.data_preprocessing.data_prepro import del_label_line, del_miss_columns, del_category_colunms, cv_delete

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


def type_identifying(data, RESULT_PATH):
    '''
        辨别变量类型
        多值有序的变量 -- mult_order
        二值变量 -- binary
        连续变量 -- quan
        多值无序的变量 -- mult_disorder
    '''
    from utils.variable_type.var_type import vars_types, var_type, var_ty
    mult_order, binary, quan, mult_disorder = vars_types(data)
    yvar_type = var_type(data)

    univariate_analysis_info = var_ty(
        data, yvar_type, mult_order, binary, quan, mult_disorder, RESULT_PATH)
    print('univariate_analysis_info: ', univariate_analysis_info)

    var_type_dict = dict(
        mult_order=mult_order,
        binary=binary,
        quan=quan,
        mult_disorder=mult_disorder,
        yvar_type=yvar_type
    )

    json.dump(var_type_dict, open(RESULT_PATH + 'logging/var_type_dict.json',
              'w', encoding='utf-8'), indent=4)
    return univariate_analysis_info, var_type_dict


def mult_disorder_to_dummies(data, var_type_dict, RESULT_PATH):
    '''
        多值无序变量设置为哑变量
    '''
    from utils.data_preprocessing.data_prepro import get_dummis
    data = get_dummis(data, var_type_dict, RESULT_PATH)

    return data


def univariate_analysis(univariate_analysis_info, RESULT_PATH):
    '''
        单因素分析：
        输出p < 0.05, 参数检验用柱状图
    '''

    from utils.univariate_analysis.univariate_quantitative import univariate_quantitative_methods, univariate_quantitative_methods1
    from utils.univariate_analysis.two_disorderly_0906 import two_disorderly
    from utils.univariate_analysis.correlation_analysis import correlation_analysis

    files = [
        'df_quant_quan_binary.xlsx',
        'df_chisquare_order_binary.xlsx',
        'df_chisquare_disorder_binary.xlsx',
        'df_chisquare_binary_binary.xlsx',
        'df_quant_quan_mult_disorder.xlsx',
        'df_chisquare_order_mult_disorder.xlsx',
        'df_chisquare_disorder_mult_disorder.xlsx',
        'df_chisquare_binary_mult_disorder.xlsx',
        'df_spe_quan.xlsx',
        'df_spe_order.xlsx',
        'df_nonparametric_disorder.xlsx',
        'df_nonparametric_binary.xlsx',
        'df_cor_quan.xlsx',
        'df_cor_order.xlsx',
        'df_quant_disorder.xlsx',
        'df_quant_binary.xlsx'
    ]

    files = [RESULT_PATH + f'tmp/{file}' for file in files]

    correlation_analysis_results = None
    stat_data = None
    stat_data_binary = None
    two_binary_results = None

    for file, t in univariate_analysis_info:
        file = RESULT_PATH + f'tmp/{file}'
        assert file in files
        result = ''
        if t == 'binary' and file == files[0]:
            result = univariate_quantitative_methods(pd.read_excel(file))
            for var_name, stat_d in result.items():
                if stat_data_binary is None:
                    stat_data_binary = stat_d
                else:
                    stat_data_binary = stat_data_binary.append(stat_d)
        elif t == 'binary' and file in files[1:4]:
            result = two_disorderly(file)
            if two_binary_results is None:
                two_binary_results = result
            else:
                two_binary_results = two_binary_results.append(result[1:])

        elif t == 'mult_disorder' and file == files[5]:
            result = univariate_quantitative_methods(pd.read_excel(file))
            for var_name, stat_data in univariate_analysis_info.items():
                output_filename = f'stat_{var_name}_p.xlsx'
                stat_data.to_excel(
                    output_filename, encoding='utf-8', index=False)
        elif t == 'mult_disorder' and file in files[6:8]:
            two_disorderly(file)
        elif t == 'mult_order' and file in files[8:10]:
            pass  # stats.spearmanr()
        elif t == 'mult_order' and file in files[10:12]:
            pass  # nonparametric_test
        elif t == 'quan' and file in files[12:14]:
            result = correlation_analysis(file)
            if correlation_analysis_results is None:
                correlation_analysis_results = result
            else:
                correlation_analysis_results = correlation_analysis_results.append(
                    result)
        elif t == 'quan' and file in files[14:16]:
            result = univariate_quantitative_methods1(pd.read_excel(file))
            for var_name, stat_d in result.items():
                if stat_data is None:
                    stat_data = stat_d
                else:
                    stat_data = stat_data.append(stat_d)
        else:
            print(file, t)

    with pd.ExcelWriter(RESULT_PATH + 'analysis/univariate_analysis.xlsx') as writer:
        if correlation_analysis_results is not None:
            correlation_analysis_results.to_excel(
                writer, encoding='utf-8', sheet_name='correlation_analysis_F1')
        if stat_data is not None:
            stat_data.to_excel(writer, encoding='utf-8',
                               index=False, sheet_name='stat_F1')
        if stat_data_binary is not None:
            stat_data_binary.to_excel(
                writer, encoding='utf-8', index=False, sheet_name='binary_stat_F1')
        if two_binary_results is not None:
            # sorting
            def sort_variable(X):
                try:
                    X = X.reset_index(drop=True)
                    indices = {v: k for k, v in X['variable']
                               [~X['variable'].isna()].to_dict().items()}
                    indices['end'] = len(X)
                    for i, k in enumerate(indices.keys()):
                        if k != 'end':
                            indices[k] = (
                                indices[k], indices[list(indices.keys())[i+1]])
                    del indices['end']
                    indices = [f'{i}:{j}' for i, j in dict(
                        sorted(indices.items(), key=lambda x: int(x[0][1:]))).values()]
                    X = X.iloc[eval(f'np.r_[0, {",".join(indices)}]'), :]
                except:
                    ...
                return X
            two_binary_results = sort_variable(two_binary_results)
            two_binary_results.to_excel(
                writer, encoding='utf-8', index=False, sheet_name='two_binary_results')


def multiple_factor_analysis(data, RESULT_PATH):
    from utils.multiple_factor_analysis.logistic import logistic
    resutls_summary = logistic(data).tables
    res1 = pd.read_html(resutls_summary[0].as_html(), header=0, index_col=0)[0]
    res2 = pd.read_html(resutls_summary[1].as_html(), header=0, index_col=0)[0]
    with pd.ExcelWriter(RESULT_PATH + 'analysis/multiple_factor_analysis.xlsx') as writer:
        res1.to_excel(writer, encoding='utf-8', sheet_name='sheet1')
        res2.to_excel(writer, encoding='utf-8', sheet_name='sheet2')


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
    from utils.data_fill.data_fil import data_impute

    data_filling_dict = {}
    for method in IMPUTE_METHODS:
        if imp_method_ is not None and method != imp_method_:
            continue
        print(f'Impute method: {method}')
        data_imp = data_impute(data, method, var_type_dict)
        data_imp.to_excel(RESULT_PATH + f'tmp/imp_{method}.xlsx', index=False)
        data_filling_dict[method] = data_imp
    return data_filling_dict


def train_test_split(data, SEED, RESULT_PATH):
    '''
        将原数据分为80%测试集和20%测试集
    '''
    from utils.split_traintest.split_train_test import train_tes_split
    df_train, df_test = train_tes_split(data, SEED)
    return df_train, df_test


def data_sampling(data, SEED, SAMPLING_METHODS, RESULT_PATH, sampling_method_=None, title=''):
    '''
        6种数据采样:
        SMO
        SSMO
        BSMO
        ADA
        ROS
        SMN
    '''
    from utils.data_sampling import data_samp
    data_sampling_dict = {}

    for sampling_method in SAMPLING_METHODS:
        if sampling_method_ is not None and sampling_method != sampling_method_:
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
    from utils.feature_selection import feature_sele

    variable_selected_dict = {}

    for sele_method in SELE_METHODS:
        if sele_method_ is not None and sele_method != sele_method_:
            continue
        print(f'Feature Selection Method: {sele_method}')
        variable_selected_dict[sele_method] = feature_sele.feature_selection(
            data, sele_method, SEED)

    return variable_selected_dict


def auto_training(data_train, data_test, SEED, model_hp_dict, TRAIN_METHODS, model_name_=None, key=''):
    from utils.model import model_pro
    from utils.model.model_analysis import auto_model

    args = model_pro.get_dataset(data_train, data_test)
    num_classes = len(set(args[2]))

    metrics_dict = {}
    errors = []

    for model_name in TRAIN_METHODS:
        if model_name_ is not None and model_name != model_name_:
            continue
        try:
            results = list(auto_model(
                args, model_hp_dict[model_name], num_classes, model_name, SEED, key).values())
            metrics_dict[model_name] = results
        except Exception as e:
            print(f'ERROR: {e}\n')
            errors.append(model_name)
    TRAIN_METHODS = [m for m in TRAIN_METHODS if m not in errors]

    return metrics_dict


def process_data(CFG, RESULT_PATH):
    data = pd.read_excel('../static/datasets/' + CFG.data_path)
    tgt_col = CFG.tgt_col
    imp_method_ = CFG.imp_method
    sampling_method_ = CFG.sampling_method
    sele_method_ = CFG.sele_method
    SEED = CFG.seed

    data = data[[tgt_col] + [col for col in data.columns if col != tgt_col]]

    # 数据清洗
    data = data_cleaning(data)

    # 判断数据类型
    univariate_analysis_info, var_type_dict = type_identifying(data)

    # 多值无序变量设置为哑变量
    data = mult_disorder_to_dummies(data, var_type_dict)

    # 数据填充6种
    data_filling_dict = data_filling(data, var_type_dict, imp_method_)

    for imp_method, data_imp in data_filling_dict.items():
        data_imp = data_filling_dict[imp_method]

        # 分为80%测试集和20%测试集
        data_imp_train, data_imp_test = train_test_split(data_imp, SEED)

        # 数据采样6种
        data_train_dict = data_sampling(
            data_imp_train, SEED, sampling_method_, 'Train Data')
        data_test_dict = data_sampling(
            data_imp_test, SEED, sampling_method_, 'Test Data')

        for sample_method in data_train_dict.keys():
            data_train, data_test = data_train_dict[sample_method], data_test_dict[sample_method]

            # 特征筛选 12种
            variable_selected_dict = feature_selection(
                data_train, SEED, sele_method_)

            for sele_method in variable_selected_dict.keys():
                vars_sele = [tgt_col] + variable_selected_dict[sele_method]
                data_train_sele = data_train[vars_sele]
                data_test_sele = data_test[vars_sele]

                key = f'{imp_method}_{sample_method}_{sele_method}'
                data_train_sele.to_csv(
                    RESULT_PATH + f'data/{key}_train.csv', encoding='utf-8-sig')
                data_test_sele.to_csv(
                    RESULT_PATH + f'data/{key}_test.csv', encoding='utf-8-sig')


def process(CFG, RESULT_PATH):
    print(RESULT_PATH)
    data = pd.read_excel('../static/datasets/' + CFG.data_path)
    tgt_col = CFG.tgt_col
    imp_method_ = CFG.imp_method
    sampling_method_ = CFG.sampling_method
    sele_method_ = CFG.sele_method
    model_name_ = CFG.model_name
    SEED = CFG.seed

    data = data[[tgt_col] + [col for col in data.columns if col != tgt_col]]

    # 数据清洗
    data = data_cleaning(data)

    # 判断数据类型
    univariate_analysis_info, var_type_dict = type_identifying(data)

    # 单因素分析
    univariate_analysis(univariate_analysis_info)

    # 多因素分析
    multiple_factor_analysis(data)

    # 多值无序变量设置为哑变量
    data = mult_disorder_to_dummies(data, var_type_dict)

    # 数据填充6种
    data_filling_dict = data_filling(data, var_type_dict, imp_method_)

    metrics_dict = {}
    import yaml
    model_hp_dict = yaml.load(
        open('../model_hp.yaml', 'r'), Loader=yaml.Loader)

    for imp_method, data_imp in data_filling_dict.items():
        data_imp = data_filling_dict[imp_method]

        # 分为80%测试集和20%测试集
        data_imp_train, data_imp_test = train_test_split(data_imp, SEED)

        # 数据采样6种
        data_train_dict = data_sampling(
            data_imp_train, SEED, sampling_method_, 'Train Data')
        data_test_dict = data_sampling(
            data_imp_test, SEED, sampling_method_, 'Test Data')

        for sample_method in data_train_dict.keys():
            data_train, data_test = data_train_dict[sample_method], data_test_dict[sample_method]

            # 特征筛选 12种
            variable_selected_dict = feature_selection(
                data_train, SEED, sele_method_)
            # json.dump(variable_selected_dict, open(
            #     RESULT_PATH + 'logging/variable_selected_dict.json', 'w', encoding='utf-8'), indent=4)

            for sele_method in variable_selected_dict.keys():
                vars_sele = [tgt_col] + variable_selected_dict[sele_method]
                data_train_sele = data_train[vars_sele]
                data_test_sele = data_test[vars_sele]

                key = f'{imp_method}_{sample_method}_{sele_method}'
                data_train_sele.to_csv(
                    RESULT_PATH + f'data/{key}_train.csv', encoding='utf-8-sig')
                data_test_sele.to_csv(
                    RESULT_PATH + f'data/{key}_test.csv', encoding='utf-8-sig')

                metrics_dict[key] = auto_training(data_train_sele, data_test_sele,
                                                  SEED, model_hp_dict, model_name_, key)

    return metrics_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# compute all other metrics and append it to the original metrics dataframe
def get_metrics(y_true, y_pred, y_score=None, return_dict=False):
    y_true = np.array(y_true).astype('int')
    y_pred = np.array(y_pred).astype('int')
    # if y_score is not None:
    #     y_score = OneHotEncoder().fit_transform(np.reshape(y_pred, (-1, 1))).toarray()
    # if len(set(y_true)) == 2 and np.ndim(y_score) > 1:
    #     y_score = y_score[:, 1]
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    brier = metrics.brier_score_loss(y_true, y_score)
    if not return_dict:
        return accuracy, precision, recall, f1_score, brier
    else:
        return dict(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            brier=brier
        )


def analyze(RESULT_PATH):
    from utils.model import model_pro
    CFG = json.load(open(RESULT_PATH + f'CONFIG.txt', 'r'))
    results_dict = json.load(
        open(RESULT_PATH + f'analysis/{CFG["out_file"]}', 'r'))
    SEED = CFG['seed']

    results = [[*key.split('_'), k, *v] for key,
               values in results_dict.items() for k, v in values.items()]
    results = np.array(results, dtype=object)
    results_auc_df = pd.DataFrame(results[:, :5], columns=[
        'Filling Method', 'Sampling Method', 'Feature Selection Method', 'Model Name', 'AUC'])
    metrics_lst = []
    for values in results:
        y_true = values[5]
        y_pred = values[6]
        y_score = values[7]

        metrics_lst.append(get_metrics(y_true, y_pred, y_score))

    metrics_df = pd.DataFrame(metrics_lst, columns=[
        'accuracy', 'precision', 'recall', 'f1_score', 'brier'])
    results_metrics_df = pd.concat((results_auc_df, metrics_df), axis=1)
    results_metrics_df.to_csv(
        RESULT_PATH + 'analysis/results_metrics.csv', index=None)

    results_auc_sort_df = results_auc_df.sort_values('AUC', ascending=False)
    best_3_mean_model_names = results_auc_df[['Model Name', 'AUC']].groupby(
        'Model Name').mean(0).sort_values('AUC', ascending=False).head(3).index.to_list()
    best_model_info = results_auc_sort_df.iloc[[0]]
    best_model_name = best_model_info['Model Name'].item()
    plot_model_names = [best_model_name] + best_3_mean_model_names
    plot_model_names = list(Counter(plot_model_names).keys())

    # plot roc curves
    plt.figure(figsize=(16, 16))
    colors = ['darkorange', 'navy', 'aqua', 'red']
    linestyles = ['--', '-', ':', '--']

    for i, plot_model_name in enumerate(plot_model_names):
        method_params = results_auc_sort_df.query(
            '`Model Name` == @plot_model_name').head(1).values.squeeze()
        # print(method_params)
        best_key, best_mn = '_'.join(method_params[:3]), method_params[3]

        auc, y_true, y_pred, y_score, params = results_dict[best_key][best_mn]

        # plot roc curve
        fpr_tpr = np.c_[metrics.roc_curve(y_true, y_score)[:2]].T
        plt.plot(fpr_tpr[0], fpr_tpr[1], linestyles[i], color=colors[i],
                 label=f'{" ".join(method_params[:4])} AUC=({auc:.2f})', lw=3, alpha=0.8)
    plt.legend()
    plt.title('ROC Curves')
    plt.savefig(RESULT_PATH + 'plots/roc_curves.pdf')
    plt.clf()

    # plot pr curves
    plt.figure(figsize=(16, 16))
    colors = ['darkorange', 'navy', 'aqua', 'red']
    linestyles = ['--', '-', ':', '--']

    for i, plot_model_name in enumerate(plot_model_names):
        method_params = results_auc_sort_df.query(
            '`Model Name` == @plot_model_name').head(1).values.squeeze()
        best_key, best_mn = '_'.join(method_params[:3]), method_params[3]

        auc, y_true, y_pred, y_score, params = results_dict[best_key][best_mn]
        # # plot PR curve
        r_p = np.c_[metrics.precision_recall_curve(y_true, y_score)[:2]].T
        plt.plot(r_p[0], r_p[1], linestyles[i], color=colors[i],
                 label=f'{" ".join(method_params[:4])} AUC=({auc:.2f})', lw=3, alpha=0.8)

    plt.legend()
    plt.title('PR Curves')
    plt.savefig(RESULT_PATH + 'plots/pr_curves.pdf')
    plt.clf()

    # plot calibration curve
    plt.figure(figsize=(16, 16))
    colors = ['darkorange', 'navy', 'aqua', 'red']
    linestyles = ['--', '-', ':', '--']

    for i, plot_model_name in enumerate(plot_model_names):
        method_params = results_auc_sort_df.query(
            '`Model Name` == @plot_model_name').head(1).values.squeeze()
        # print(method_params)
        best_key, best_mn = '_'.join(method_params[:3]), method_params[3]

        auc, y_true, y_pred, y_score, params = results_dict[best_key][best_mn]

        # plot roc curve
        fpr_tpr = np.c_[metrics.roc_curve(y_true, y_score)[:2]].T
        plt.plot(fpr_tpr[0], fpr_tpr[1], linestyles[i], color=colors[i],
                 label=f'{" ".join(method_params[:4])} AUC=({auc:.2f})', lw=3, alpha=0.8)
    plt.legend()
    plt.title('Calibration Curves')
    plt.savefig(RESULT_PATH + 'plots/calibration_curve.pdf')
    plt.clf()

    data = pd.read_excel(f'../static/datasets/{CFG["data_path"]}')
    tgt_col = CFG['tgt_col']

    fi, sa, se, mn = best_model_info.values.squeeze()[:4]

    var_type_dict = json.load(
        open(RESULT_PATH + f'logging/var_type_dict.json', 'r'))
    columns = [k for key in var_type_dict if key in ['mult_order',
                                                     'binary', 'quan', 'mult_disorder'] for k in var_type_dict[key]]
    data = data[[tgt_col] + columns]
    print(data.columns)
    data = data_filling(data, var_type_dict, fi)[fi]
    data_train, data_test = train_test_split(data, SEED)

    data_train = data_sampling(data_train, SEED, sa, 'Train Data')[sa]
    data_test = data_sampling(data_test, SEED, sa, 'Test Data')[sa]

    variable_selected = feature_selection(data_train, SEED, se)[se]

    data_train = data_train[[tgt_col] + variable_selected]
    data_test = data_test[[tgt_col] + variable_selected]

    args = model_pro.get_dataset(data_train, data_test)
    X_train, X_test, y_train, y_test = args

    args = model_pro.get_dataset(data_train, data_test)
    X_train, X_test, y_train, y_train = args

    X = data_train.copy()
    y = X.pop(tgt_col)

    model_hp_dict = yaml.load(
        open('../model_hp.yaml', 'r'), Loader=yaml.Loader)
    key = f'{fi}_{sa}_{se}'
    params = results_dict[key][mn][4]
    try:
        model = eval(model_hp_dict[mn][0])(**params, random_state=SEED)
    except:
        model = eval(model_hp_dict[mn][0])(**params)
    model.fit(X, y)

    y_true = y_test.values
    y_pred = model.predict(X_test)
    try:
        y_score = model._predict_proba_lr(X_test)
    except:
        y_score = model.predict_proba(X_test)
    if len(set(y_true)) == 2:
        y_score = y_score[:, 1]

    try:
        explainer = shap.Explainer(model)
    except:
        masker = shap.maskers.Independent(X)
        try:
            explainer = shap.Explainer(model, masker)
        except:
            print('Cannot Plot SHAP')
            return
    shap_values = explainer(X)

    try:
        shap.summary_plot(shap_values, X, plot_type='bar', show=False)
        plt.gcf().savefig(RESULT_PATH + f'plots/{key}_summary_plot.pdf')
        print('Plotting Summary Plot')
        plt.clf()
    except Exception as e:
        print(f'Beesarm Plot Error: {e}')

    try:
        try:
            shap_values.values
            tmp = shap.Explanation(shap_values, data=X,
                                   feature_names=X.columns)
        except:
            tmp = shap.Explanation(
                shap_values[:, :, 1], data=X, feature_names=X.columns)
        shap.plots.beeswarm(tmp, show=False, color_bar=True,
                            plot_size=(12, 9), max_display=10000)
        plt.gcf().savefig(RESULT_PATH + f'plots/{key}_beeswarm.pdf')
        print('Plotting Beesarm Plot')
        plt.clf()
    except Exception as e:
        print(f'Beesarm Plot Error: {e}')

    try:
        ax = shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X.iloc[0],
                             feature_names=X.columns, show=False, matplotlib=True,
                             )
        ax.savefig(RESULT_PATH +
                   f'plots/{key}_force_plot.pdf', bbox_inches='tight')
        print('Plotting Force Plot')
        plt.clf()
    except Exception as e:
        print(f'Force Plot Error: {e}')


def runner(CFG):
    seed_everything(CFG.seed)

    sys.path.append('..')

    '''
        抽样方法：
        SMO, SSMO, BSMO, ADA, ROS, SMN, 
    '''
    SAMPLING_METHODS = ['SMO', 'SSMO', 'BSMO', 'ADA', 'ROS', 'SMN']
    IMPUTE_METHODS = ['Simple', 'KNN', 'ISVD', 'Imput', 'rf', 'optimal']
    SELE_METHODS = ['Las', 'RCV', 'ENC', 'Cat', 'SVC', 'RF',
                    'Ada', 'GBC', 'ExT', 'BNB', 'XGB', 'LGBM']
    TRAIN_METHODS = ['Multinomial Naive Bayes', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'Extra Tree', 'SVC',
                     'Bernoulli Naive Bayes', 'XGBoost', 'LGBoost', 'Gaussian Naive Bayes',
                     'Complement Naive Bayes', 'KNN', 'CatBoost',
                     'Decision Tree', 'QDA', 'Passive Aggressive', 'LDA', 'Logistic Regression',
                     'SGD', 'Bagging', 'MLP']

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None,
                        help='Data Filename in the Datasets Folder')
    parser.add_argument('--tgt_col', default=None,
                        help='Target Column Name')
    parser.add_argument('--imp_method', default=None,
                        help='Data Filling Method')
    parser.add_argument('--sampling_method', default=None,
                        help='Data Sampling Method')
    parser.add_argument('--sele_method', default=None,
                        help='Feature Selection Method')
    parser.add_argument('--model_name', default=None,
                        help='Training Model Name')
    parser.add_argument('--out_file', default='metrics.json',
                        help='Output Filename in the results Folder')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random Seed')
    parser.add_argument('--not_analyze', action='store_true')

    CFG = parser.parse_args()
    if CFG.seed is None:
        CFG.seed = np.random.randint(0, 99999999)

    while CFG.data_path is None:
        dataset_name = input('在static/datasets文件夹中的数据集名称: ')
        if not os.path.exists(f'../static/datasets/{dataset_name}'):
            CFG.data_path = None
            print(f'{dataset_name} 不在static/datasets文件夹中')
        else:
            CFG.data_path = dataset_name

    columns_ = pd.read_excel(
        f'../static/datasets/{CFG.data_path}', skipfooter=99999).columns
    while CFG.tgt_col is None:
        y_col = input(f'数据集{CFG.data_path}中标签列的表头名: ')
        if y_col not in columns_:
            CFG.tgt_col = None
            print(f'表头名{y_col}不在数据集{CFG.data_path}中')
        else:
            CFG.tgt_col = y_col

    print(CFG.data_path, CFG.seed)

    data_name, fmt = CFG.data_path.split('.')

    RESULT_PATH = f'../results/{data_name}/'
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(RESULT_PATH + 'tmp', exist_ok=True)
    os.makedirs(RESULT_PATH + 'logging', exist_ok=True)
    os.makedirs(RESULT_PATH + 'analysis', exist_ok=True)
    os.makedirs(RESULT_PATH + 'data', exist_ok=True)

    json.dump(vars(CFG), open(RESULT_PATH + 'CONFIG.txt', 'w'), indent=4)
    CFG.RESULT_PATH = RESULT_PATH

    try:
        metrics_dict = process_data(CFG)
        # json.dump(metrics_dict, open(
        # RESULT_PATH + f'analysis/{CFG.out_file}', 'w', encoding='utf-8'), indent=4, cls=NpEncoder)
    except Exception as e:
        print(traceback.format_exc())

    if not CFG.not_analyze:
        os.makedirs(RESULT_PATH + 'plots', exist_ok=True)
        analyze(RESULT_PATH)

    try:
        shutil.rmtree(RESULT_PATH + 'tmp')
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    # print('EOF')
