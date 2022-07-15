import os
from django.conf import settings
import numpy as np
import pandas as pd
from sklearn import metrics

from rpy2 import robjects
import traceback
from collections import Counter
from matplotlib import pyplot as plt
import json

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

SAVE_FORMAT = ['.png', '.pdf']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def var_ty(df, yvar_type, var_type_dict, RESULT_PATH):
    mult_order = var_type_dict['mult_order']
    binary = var_type_dict['binary']
    quan = var_type_dict['quan']
    mult_disorder = var_type_dict['mult_disorder']

    res = []
    if yvar_type == 'binary':
        if len(quan) > 0:
            df_quant_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_quant_quan.to_excel(RESULT_PATH + f'tmp/' +
                                   "df_quant_quan_binary.xlsx", index=False)
            res.append(("df_quant_quan_binary.xlsx", 'binary'))
        if len(mult_order) > 0:
            df_chi_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_chi_order.to_excel(RESULT_PATH + f'tmp/' +
                                  "df_chisquare_order_binary.xlsx", index=False)
            res.append(("df_chisquare_order_binary.xlsx", 'binary'))
        if len(mult_disorder) > 0:
            df_chi_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_chi_disorder.to_excel(RESULT_PATH + f'tmp/' +
                                     "df_chisquare_disorder_binary.xlsx", index=False)
            res.append(("df_chisquare_disorder_binary.xlsx", 'binary'))
        if len(binary) > 0:
            df_chi_binary = pd.concat([df.iloc[:, 0], df[binary]], axis=1)
            df_chi_binary.to_excel(RESULT_PATH + f'tmp/' +
                                   "df_chisquare_binary_binary.xlsx", index=False)
            res.append(("df_chisquare_binary_binary.xlsx", 'binary'))
    elif yvar_type == 'mult_disorder':
        if len(quan) > 0:
            df_quant_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_quant_quan.to_excel(RESULT_PATH + f'tmp/' +
                                   "df_quant_quan_mult_disorder.xlsx", index=False)
            res.append(("df_quant_quan_mult_disorder.xlsx", 'mult_disorder'))
        if len(mult_order) > 0:
            df_chi_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_chi_order.to_excel(RESULT_PATH + f'tmp/' +
                                  "df_chisquare_order_mult_disorder.xlsx", index=False)
            res.append(
                ("df_chisquare_order_mult_disorder.xlsx", 'mult_disorder'))
        if len(mult_disorder) > 0:
            df_chi_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_chi_disorder.to_excel(RESULT_PATH + f'tmp/' +
                                     "df_chisquare_disorder_mult_disorder.xlsx", index=False)
            res.append(
                ("df_chisquare_disorder_mult_disorder.xlsx", 'mult_disorder'))
        if len(binary) > 0:
            df_chi_binary = pd.concat([df.iloc[:, 0], df[binary]], axis=1)
            df_chi_binary.to_excel(RESULT_PATH + f'tmp/' +
                                   "df_chisquare_binary_mult_disorder.xlsx", index=False)
            res.append(
                ("df_chisquare_binary_mult_disorder.xlsx", 'mult_disorder'))
    elif yvar_type == 'mult_order':
        if len(quan) > 0:
            df_spe_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)

            df_spe_quan.to_excel(RESULT_PATH + f'tmp/' +
                                 "df_spe_quan.xlsx", index=False)
            res.append(("df_spe_quan.xlsx", 'mult_order'))

        if len(mult_order) > 0:
            df_spe_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_spe_order.to_excel(RESULT_PATH + f'tmp/' +
                                  "df_spe_order.xlsx", index=False)
            res.append(("df_spe_order.xlsx", 'mult_order'))

        if len(mult_disorder) > 0:
            df_nonparametric_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_nonparametric_disorder.to_excel(RESULT_PATH + f'tmp/' +
                                               "df_nonparametric_disorder.xlsx", index=False)
            res.append(("df_nonparametric_disorder.xlsx", 'mult_order'))

        if len(binary) > 0:
            df_nonparametric_binary = pd.concat(
                [df.iloc[:, 0], df[binary]], axis=1)
            df_nonparametric_binary.to_excel(RESULT_PATH + f'tmp/' +
                                             "df_nonparametric_binary.xlsx", index=False)
            res.append(("df_nonparametric_binary.xlsx", 'mult_order'))
    elif yvar_type == 'quan':
        if len(quan) > 0:
            df_cor_quan = pd.concat([df.iloc[:, 0], df[quan]], axis=1)
            df_cor_quan.to_excel(RESULT_PATH + f'tmp/' +
                                 "df_cor_quan.xlsx", index=False)
            res.append(("df_cor_quan.xlsx", 'quan'))

        if len(mult_order) > 0:
            df_quant_order = pd.concat([df.iloc[:, 0], df[mult_order]], axis=1)
            df_quant_order.to_excel(
                RESULT_PATH + f'tmp/' + "df_cor_order.xlsx", index=False)
            res.append(("df_cor_order.xlsx", 'quan'))

        if len(mult_disorder) > 0:
            df_quant_disorder = pd.concat(
                [df.iloc[:, 0], df[mult_disorder]], axis=1)
            df_quant_disorder.to_excel(
                RESULT_PATH + f'tmp/' + "df_quant_disorder.xlsx", index=False)
            res.append(("df_quant_disorder.xlsx", 'quan'))

        if len(binary) > 0:
            df_quant_binary = pd.concat([df.iloc[:, 0], df[binary]], axis=1)
            df_quant_binary.to_excel(
                RESULT_PATH + f'tmp/' + "df_quant_binary.xlsx", index=False)
            res.append(("df_quant_binary.xlsx", 'quan'))
    return res


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


def analyze(metrics_dict, RESULT_PATH):

    results = [[*key.split('_'), k, *v] for key,
               values in metrics_dict.items() for k, v in values.items()]
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
    results_metrics_df = results_metrics_df.sort_values(
        'AUC', ascending=False).fillna('')
    results_metrics_df.to_csv(
        RESULT_PATH + f'analysis/results_metrics.csv', index=None)
    return RESULT_PATH + f'analysis/results_metrics.csv'


def get_plots(CFG, results_metrics_df, metrics_dict, RESULT_PATH):
    os.makedirs(RESULT_PATH + f'plots', exist_ok=True)

    best_3_mean_model_names = results_metrics_df[['Model Name', 'AUC']].groupby(
        'Model Name').mean(0).sort_values('AUC', ascending=False).head(3).index.to_list()
    best_model_info = results_metrics_df.iloc[[0]]
    best_model_name = best_model_info['Model Name'].item()
    plot_model_names = [best_model_name] + best_3_mean_model_names
    plot_model_names = list(Counter(plot_model_names).keys())

    # plot roc curves
    plt.figure(figsize=(16, 16))
    colors = ['darkorange', 'navy', 'aqua', 'red']
    linestyles = ['--', '-', ':', '--']

    for i, plot_model_name in enumerate(plot_model_names):
        method_params = results_metrics_df.query(
            '`Model Name` == @plot_model_name').head(1).values.squeeze()
        # print(method_params)
        best_key, best_mn = '_'.join(method_params[:3]), method_params[3]

        auc, y_true, y_pred, y_score, params = metrics_dict[best_key][best_mn]

        # plot roc curve
        fpr_tpr = np.c_[metrics.roc_curve(y_true, y_score)[:2]].T
        plt.plot(fpr_tpr[0], fpr_tpr[1], linestyles[i], color=colors[i],
                 label=f'{" ".join(method_params[:4])} AUC=({auc:.2f})', lw=3, alpha=0.8)
    plt.legend()
    plt.title('ROC Curves')
    for fmt in SAVE_FORMAT:
        plt.savefig(
            RESULT_PATH + f'plots/roc_curves{fmt}', bbox_inches='tight')
    plt.clf()

    # plot pr curves
    plt.figure(figsize=(16, 16))
    colors = ['darkorange', 'navy', 'aqua', 'red']
    linestyles = ['--', '-', ':', '--']

    for i, plot_model_name in enumerate(plot_model_names):
        method_params = results_metrics_df.query(
            '`Model Name` == @plot_model_name').head(1).values.squeeze()
        best_key, best_mn = '_'.join(method_params[:3]), method_params[3]

        auc, y_true, y_pred, y_score, params = metrics_dict[best_key][best_mn]
        # # plot PR curve
        r_p = np.c_[metrics.precision_recall_curve(y_true, y_score)[:2]].T
        plt.plot(r_p[0], r_p[1], linestyles[i], color=colors[i],
                 label=f'{" ".join(method_params[:4])} AUC=({auc:.2f})', lw=3, alpha=0.8)

    plt.legend()
    plt.title('PR Curves')
    for fmt in SAVE_FORMAT:
        plt.savefig(
            RESULT_PATH + f'plots/pr_curves{fmt}', bbox_inches='tight')
    plt.clf()

    # plot calibration curve
    plt.figure(figsize=(16, 16))
    colors = ['darkorange', 'navy', 'aqua', 'red']
    linestyles = ['--', '-', ':', '--']

    for i, plot_model_name in enumerate(plot_model_names):
        method_params = results_metrics_df.query(
            '`Model Name` == @plot_model_name').head(1).values.squeeze()
        # print(method_params)
        best_key, best_mn = '_'.join(method_params[:3]), method_params[3]

        auc, y_true, y_pred, y_score, params = metrics_dict[best_key][best_mn]

        # plot roc curve
        fpr_tpr = np.c_[metrics.roc_curve(y_true, y_score)[:2]].T
        plt.plot(fpr_tpr[0], fpr_tpr[1], linestyles[i], color=colors[i],
                 label=f'{" ".join(method_params[:4])} AUC=({auc:.2f})', lw=3, alpha=0.8)
    plt.legend()
    plt.title('Calibration Curves')
    for fmt in SAVE_FORMAT:
        plt.savefig(
            RESULT_PATH + f'plots/calibration_curve{fmt}', bbox_inches='tight')
    plt.clf()

    # file_path = os.path.join(MEDIA_ROOT, CFG['data_name'])
    # try:
    #     data = pd.read_excel(file_path + '.xlsx', nrows=30)
    # except:
    #     data = pd.read_csv(file_path + '.csv', nrows=30)

    tgt_col = CFG['tgt_col']

    fi, sa, se, mn = best_model_info.values.squeeze()[:4]
    key = f'{fi}_{sa}_{se}'
    train_data = pd.read_csv(f'{RESULT_PATH}data/{key}_train.csv')
    test_data = pd.read_csv(f'{RESULT_PATH}data/{key}_test.csv')

    var_type_dict = json.load(
        open(RESULT_PATH + f'logging/var_type_dict.json', 'r'))
    columns = [k for key in var_type_dict if key in ['mult_order',
                                                     'binary', 'quan', 'mult_disorder'] for k in var_type_dict[key]]
    print(train_data.columns)

    train_X = train_data.copy()
    train_y = train_X.pop(tgt_col)
    test_X = test_data.copy()
    test_y = test_X.pop(tgt_col)

    data_name = CFG['data_name']

    os.system(f'python app/core/dca_curve.py {data_name}')

    SEED = CFG['seed']
    import yaml
    model_hp_dict = yaml.load(
        open('model_hp.yaml', 'r'), Loader=yaml.Loader)
    params = metrics_dict[key][mn][4]
    try:
        model = eval(model_hp_dict[mn][0])(**params, random_state=SEED)
    except:
        model = eval(model_hp_dict[mn][0])(**params)
    model.fit(train_X, train_y)

    y_true = test_y.values
    y_pred = model.predict(test_X)
    try:
        y_score = model._predict_proba_lr(test_X)
    except:
        y_score = model.predict_proba(test_X)
    if len(set(y_true)) == 2:
        y_score = y_score[:, 1]
    import shap
    try:
        explainer = shap.Explainer(model)
    except:
        masker = shap.maskers.Independent(test_X)
        try:
            explainer = shap.Explainer(model, masker)
        except Exception as e:
            print('Cannot Plot SHAP.', e)
            return RESULT_PATH + f'plots'
    shap_values = explainer(test_X)

    try:
        shap.summary_plot(shap_values, test_X, plot_type='bar', show=False)
        for fmt in SAVE_FORMAT:
            plt.gcf().savefig(RESULT_PATH +
                              f'plots/{key}_summary_plot{fmt}', bbox_inches='tight')
        print('Plotting Summary Plot')
        plt.clf()
    except Exception as e:
        print(f'Beesarm Plot Error: {e}')

    try:
        try:
            shap_values.values
            tmp = shap.Explanation(shap_values, data=test_X,
                                   feature_names=test_X.columns)
        except:
            tmp = shap.Explanation(
                shap_values[:, :, 1], data=test_X, feature_names=test_X.columns)
        shap.plots.beeswarm(tmp, show=False, color_bar=True,
                            plot_size=(12, 9), max_display=10000)
        for fmt in SAVE_FORMAT:
            plt.gcf().savefig(RESULT_PATH +
                              f'plots/{key}_beeswarm{fmt}', bbox_inches='tight')
        print('Plotting Beesarm Plot')
        plt.clf()
    except Exception as e:
        print(f'Beesarm Plot Error: {e}')

    try:
        ax = shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], test_X.iloc[0],
                             feature_names=test_X.columns, show=False, matplotlib=True,
                             )
        for fmt in SAVE_FORMAT:
            ax.savefig(RESULT_PATH +
                       f'plots/{key}_force_plot{fmt}', bbox_inches='tight')
        print('Plotting Force Plot')
        plt.clf()
    except Exception as e:
        print(f'Force Plot Error: {e}')

    return RESULT_PATH + f'plots'


def analyze_metrics_resuls(data_name, RESULT_PATH):
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as sm
    from .utils.univariate_analysis.univariate_quantitative import univariate_quantitative_methods1

    import warnings
    warnings.filterwarnings('ignore')
    columns_name = ['variable', 'group', 'Mean±SD',
                    'statistics', 'P values', 'method']

    df = pd.read_csv(f'results/{data_name}/analysis/results_metrics.csv')

    independent_vars = [var for var in [
        'Filling Method', 'Sampling Method', 'Feature Selection Method', 'Model Name']
        if df[var].nunique() > 1]
    dependent_vars = ['AUC', 'accuracy', 'precision', 'recall', 'f1_score']

    with pd.ExcelWriter(RESULT_PATH + 'analysis/results_metrics_analysis.xlsx') as writer:
        for dependent_var in dependent_vars:
            prco_data = df[[dependent_var] + independent_vars]
            univariate_quantitative_methods1(prco_data)[dependent_var].fillna('').to_excel(
                writer, encoding='utf-8', index=False, sheet_name=f'{dependent_var}-单因素分析')

            prco_data.columns = [col.replace(' ', '_')
                                 for col in prco_data.columns]
            model = sm.ols(
                f"{prco_data.columns[0]}~{'+'.join(prco_data.columns[1:])}", data=prco_data).fit()
            for i, table in enumerate([pd.read_html(res.as_html(), header=0, index_col=0)[0] for res in model.summary().tables]):
                table.to_excel(writer, encoding='utf-8', index=True,
                               sheet_name=f'{dependent_var}-多因素分析-{i+1}')
    return RESULT_PATH + 'analysis/results_metrics_analysis.xlsx'
