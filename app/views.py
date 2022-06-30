import os
import pandas as pd
import json
import shutil
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseNotFound, HttpResponse, HttpResponseRedirect
from django.core.files.storage import FileSystemStorage

from django.urls import reverse
from .forms import ConfigForm
from .storage import OverwriteStorage
from .settings import MEDIA_ROOT

from .models import Config

from .core.process import preprocess_runner, analysis_runner, model_runner
from .core.analysis import analyze, get_plots, NpEncoder


NAN = -999.99999999


def get_config(data_name, seed=20):
    CFG = list(Config.objects.all().filter(data_name=data_name))[-1].__dict__
    CFG['variables'] = CFG['variables'].split('||')
    CFG['var_types'] = CFG['var_types'].split('|')
    CFG['imp_method'] = CFG['imp_method'].split('|')
    CFG['sampling_method'] = CFG['sampling_method'].split('|')
    CFG['sele_method'] = CFG['sele_method'].split('|')
    CFG['model_name'] = CFG['model_name'].split('|')
    CFG['seed'] = seed
    return CFG


def index(request, data_name=None):

    context = {}
    context['sections'] = 1
    context['display_table'] = False

    if request.method == 'POST' and request.FILES['file_data'] and 'show' in request.POST:
        data = request.FILES['file_data']
        os.makedirs(MEDIA_ROOT, exist_ok=True)
        file_path = os.path.join(MEDIA_ROOT, data.name)

        if os.path.exists(file_path):
            os.remove(os.path.join(MEDIA_ROOT, data.name))

        fs = FileSystemStorage()
        file_name = fs.save(data.name, data)
        data_name = file_name.split('.')[0]
        return HttpResponseRedirect(data_name, data_name)

    if data_name is not None:
        file_path = os.path.join(MEDIA_ROOT, data_name)
        try:
            data_df = pd.read_excel(file_path + '.xlsx', nrows=30)
            file_name = data_name + '.xlsx'
        except:
            try:
                data_df = pd.read_csv(file_path + '.csv', nrows=30)
                file_name = data_name + '.csv'
            except:
                return render(request, 'index.html', context)

        data_df = data_df.fillna(NAN).round(3)
        data_df = data_df.replace(NAN, '')
        os.makedirs(f'results/{data_name}', exist_ok=True)
        context['data_name'] = data_name
        context['file_name'] = file_name
        context['file_cols'] = [col if col[:7] != 'Unnamed' else '' for col in data_df.columns]
        context['file_data'] = data_df.values.tolist()
        context['display_table'] = True
        return render(request, 'index.html', context)

    return render(request, 'index.html', context)


def config(request, data_name=None):

    file_path = os.path.join(MEDIA_ROOT, data_name)
    try:
        data_df = pd.read_excel(file_path + '.xlsx', nrows=30)
    except:
        data_df = pd.read_csv(file_path + '.csv', nrows=30)

    form_config = dict(
        variables=list(data_df.columns),
        data_df=data_df
    )

    if request.method == 'POST':
        form = ConfigForm(request.POST, **form_config)
        if form.is_valid():

            form_data = form.cleaned_data
            var_type_dict = {key[4:]: form_data[key]
                             for key in form_data if key[:4] == 'var-' and form_data[key] != 'delete'}

            save_form = Config(
                data_name=data_name,
                tgt_col=form_data['Target Column'],
                variables='||'.join(list(var_type_dict.keys())),
                var_types='|'.join(list(var_type_dict.values())),
                imp_method='|'.join(form_data['Impute Methods']),
                sampling_method='|'.join(form_data['Sampling Methods']),
                sele_method='|'.join(form_data['Selection Methods']),
                model_name='|'.join(form_data['Train Methods']),
            )
            save_form.save()
            return HttpResponseRedirect(f'/{data_name}/preprocess')
    else:
        form = ConfigForm(**form_config)

    return render(request, 'config.html', {'form': form, 'data_name': data_name})


def preprocess(request, data_name=None):

    CFG = get_config(data_name)
    CFG, RESULT_PATH, var_type_dict = preprocess_runner(CFG)
    print(CFG)
    print(RESULT_PATH)
    print(var_type_dict)
    result_path = analysis_runner(CFG, var_type_dict, RESULT_PATH)

    datum = pd.ExcelFile(result_path)
    sheet_names = datum.sheet_names
    data_cols = []
    data_values = []
    for sn in sheet_names:
        data = pd.read_excel(datum, sn).fillna('')
        data_cols.append(
            [col if col[:7] != 'Unnamed' else '' for col in data.columns])
        data_values.append(data.values.tolist())

    return render(request, 'preprocess.html', {
        'sheet_ids': list(range(len(sheet_names))),
        'sheet_data': list(zip(sheet_names, data_cols, data_values)),
        'data_name': data_name
    })


def process(request, data_name=None):
    CFG = get_config(data_name)
    RESULT_PATH = f'results/{data_name}/'
    results_path = RESULT_PATH + 'analysis/results_metrics.csv'
    try:
        results = pd.read_csv(results_path).round(2)
    except:
        CFG['RESULT_PATH'] = RESULT_PATH
        metrics_dict = model_runner(CFG)
        import json
        json.dump(metrics_dict, open(
            RESULT_PATH + f'analysis/metrics_dict.json', 'w', encoding='utf-8'), indent=4, cls=NpEncoder)
        results_path = analyze(metrics_dict, RESULT_PATH)
        results = pd.read_csv(results_path).round(2)

    return render(request, 'process.html', {
        'data_name': data_name,
        'file_cols': list(results.columns),
        'file_data': results.values.tolist()
    })


def plots(request, data_name=None):
    CFG = get_config(data_name)
    RESULT_PATH = f'results/{data_name}/'
    results_path = RESULT_PATH + 'analysis/results_metrics.csv'
    results_metrics_df = pd.read_csv(results_path)
    metrics_dict = json.load(
        open(RESULT_PATH + f'analysis/metrics_dict.json', 'r'))
    plot_path = get_plots(CFG, results_metrics_df, metrics_dict, RESULT_PATH)    
    # img_paths = [f'.{plot_path}/{file}' for file in os.listdir(plot_path)]
    files = os.listdir(plot_path)
    img_paths = [f'{plot_path}/{file}' for file in files]
    
    for img_path, file in zip(img_paths, files):
        shutil.copy(img_path, os.path.join(MEDIA_ROOT, f'{data_name}_{file}'))
    
    img_paths = [f'/media/{data_name}_{file}' for file in files]
    return render(request, 'plots.html', {
        'data_name': data_name,
        'img_paths': img_paths
    })
    

# results/定性资料-糖化蛋白/plots/roc_curves.png
# /results/定性资料-糖化蛋白/plots/pr_curves.png