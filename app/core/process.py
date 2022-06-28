from .. import settings

TRAIN_METHODS = settings.TRAIN_METHODS[1].keys()

def auto_training(train_data, test_data, SEED, model_name_=None, key='', HP_PATH='model_hp.yaml'):
    import yaml
    from .utils.model import model_pro
    from .utils.model.model_analysis import auto_model
    import os
    
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
