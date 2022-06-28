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

from .. import settings

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
        try:
            results = list(auto_model(
                args, model_hp_dict[model_name], num_classes, model_name, SEED, key).values())
            metrics_dict[model_name] = results
        except Exception as e:
            print(f'ERROR: {e}\n')
            metrics_dict[model_name] = None

    return metrics_dict


def runner(CFG):
    os.makedirs('results', exist_ok=True)
    data_name = CFG['data_name']
    RESULT_PATH = f'results/{data_name}/'
    import pandas as pd
    try:
        data = pd.read_excel(f'{settings.MEDIA_ROOT}/{data_name}.xlsx')
    except:
        data = pd.read_csv(f'{settings.MEDIA_ROOT}/{data_name}.csv')

    try:
        shutil.rmtree(RESULT_PATH)
    except:
        ...

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
    return RESULT_PATH, process_data(CFG, RESULT_PATH)
