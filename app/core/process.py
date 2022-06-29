from .analysis import var_ty
from .. import settings
import codecs
from .preprocessing import (
    seed_everything,
    process_data
)
import shutil
import json
import os

import warnings
warnings.filterwarnings('ignore')


TRAIN_METHODS = settings.TRAIN_METHODS[1].keys()


def auto_training(train_data, test_data, SEED, model_name_=None, key='', HP_PATH='model_hp.yaml'):
    import yaml
    from .utils.model import model_pro
    from .utils.model.model_analysis import auto_model

    model_hp_dict = yaml.load(open(HP_PATH, 'r'), Loader=yaml.Loader)
    args = model_pro.get_dataset(train_data, test_data)
    num_classes = len(set(args[2]))

    metrics_dict = {}

    for model_name in TRAIN_METHODS:
        if model_name_ is not None and model_name not in model_name_:
            continue

        results = list(auto_model(
            args, model_hp_dict[model_name], num_classes, model_name, SEED, key).values())
        metrics_dict[model_name] = results

    return metrics_dict


def preprocess_runner(CFG):
    os.makedirs('results', exist_ok=True)
    data_name = CFG['data_name']
    RESULT_PATH = f'results/{data_name}/'
    import pandas as pd
    try:
        data = pd.read_excel(f'{settings.MEDIA_ROOT}/{data_name}.xlsx')
    except:
        data = pd.read_csv(f'{settings.MEDIA_ROOT}/{data_name}.csv')

    shutil.rmtree(RESULT_PATH)

    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(RESULT_PATH + 'tmp', exist_ok=True)
    os.makedirs(RESULT_PATH + 'logging', exist_ok=True)
    os.makedirs(RESULT_PATH + 'analysis', exist_ok=True)
    os.makedirs(RESULT_PATH + 'data', exist_ok=True)
    json.dump({key: CFG[key] for key in CFG if key[0] != '_'},
              codecs.open(RESULT_PATH + 'CONFIG.txt', 'w'), indent=4, ensure_ascii=False)
    seed_everything(CFG['seed'])
    CFG['data'] = data
    print(CFG)
    return CFG, RESULT_PATH, process_data(CFG, RESULT_PATH)


def analysis_runner(CFG, var_type_dict, RESULT_PATH):

    import pandas as pd
    from scipy import stats

    from statsmodels.genmod.families import Binomial

    from .utils.univariate_analysis.univariate_quantitative import univariate_quantitative_methods, univariate_quantitative_methods1
    from .utils.univariate_analysis.two_disorderly_0906 import two_disorderly
    from .utils.univariate_analysis.correlation_analysis import correlation_analysis
    from .utils.univariate_analysis.qualitaive_analysis import nonparametric_test

    yvar_type = CFG['var_types'][CFG['variables'].index(CFG['tgt_col'])]
    for key in var_type_dict:
        var_type_dict[key] = [
            item for item in var_type_dict[key] if item != CFG['tgt_col']]
    json.dump(var_type_dict, open(RESULT_PATH + 'logging/var_type_dict.json',
                                  'w', encoding='utf-8'), indent=4)
    res = var_ty(CFG['data'], yvar_type, var_type_dict, RESULT_PATH)
    data = CFG['data']

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

    files = [RESULT_PATH + 'tmp/' + file for file in files]

    correlation_analysis_results = None
    stat_data = None
    stat_data_binary = None
    two_binary_results = None

    for file, t in res:
        file = RESULT_PATH + 'tmp/' + file
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

        elif t == 'mult_disorder' and file == files[4]:
            result = univariate_quantitative_methods(pd.read_excel(file))
            for var_name, stat_data in result.items():
                output_filename = f'stat_{var_name}_p.xlsx'
                stat_data.to_excel(
                    output_filename, encoding='utf-8', index=False)
        elif t == 'mult_disorder' and file in files[5:8]:
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
            raise

    def process_statsmodels(data, model_name, title):
        import statsmodels.formula.api as sm
        model = eval(f'sm.{model_name}')(f"{data.columns[0]}~{'+'.join(data.columns[1:])}",
                                         data=data, family=Binomial() if model_name == 'glm' else None).fit()
        # with pd.ExcelWriter(RESULT_PATH + f'{title}.xlsx') as writer:
        #     for i, res in enumerate(model.summary().tables):
        #         pd.read_html(res.as_html(), header=0, index_col=0)[0].to_excel(
        #             writer, encoding='utf-8', sheet_name=f'sheet{i+1}')
        return [pd.read_html(res.as_html(), header=0, index_col=0)[0] for res in model.summary().tables]

    # 判断定性还是定量
    # print('判断定性还是定量', yvar_type)

    with pd.ExcelWriter(RESULT_PATH + 'analysis/results.xlsx') as writer:
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
            two_binary_results.to_excel(
                writer, encoding='utf-8', index=False, sheet_name='two_binary_results')

        dfs = []
        if yvar_type == 'binary':
            dfs += process_statsmodels(data, 'glm', 'logistic')

        elif yvar_type == 'quan':
            dfs += process_statsmodels(data, 'ols', 'multiple_analysis')

        for i, df in enumerate(dfs):
            sheet_name = f'sheet{i + 1}'
            if i == 1 and yvar_type == 'binary':
                sheet_name = 'logistic回归'
            if i == 1 and yvar_type == 'quan':
                sheet_name = '多重线性回归'

            df.to_excel(writer, encoding='utf-8',
                        index=True, sheet_name=sheet_name)

    return RESULT_PATH + 'analysis/results.xlsx'


def model_runner(CFG):
    import pandas as pd
    RESULT_PATH = CFG['RESULT_PATH']
    all_keys = set([key.rsplit('_', 1)[0]
                   for key in os.listdir(RESULT_PATH + 'data')])
    metrics_dict = {}
    for key in all_keys:
        train_data = pd.read_csv(f'{RESULT_PATH}data/{key}_train.csv')
        test_data = pd.read_csv(f'{RESULT_PATH}data/{key}_test.csv')
        metrics_dict[key] = auto_training(train_data, test_data, CFG['seed'],
                                          model_name_=CFG['model_name'], key=key,
                                          HP_PATH='model_hp.yaml')
    return metrics_dict
