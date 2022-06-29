import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseNotFound, HttpResponse, HttpResponseRedirect
from django.core.files.storage import FileSystemStorage

from django.urls import reverse
from .forms import ConfigForm
from .storage import OverwriteStorage
from .settings import MEDIA_ROOT

from .models import Config

from .core.process import preprocess_runner, analysis_runner


NAN = -999.99999999


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
        context['data_name'] = data_name
        context['file_name'] = file_name
        context['file_cols'] = list(data_df.columns)
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

    CFG = list(Config.objects.all().filter(data_name=data_name))[-1].__dict__
    CFG['variables'] = CFG['variables'].split('||')
    CFG['var_types'] = CFG['var_types'].split('|')
    CFG['imp_method'] = CFG['imp_method'].split('|')
    CFG['sampling_method'] = CFG['sampling_method'].split('|')
    CFG['sele_method'] = CFG['sele_method'].split('|')
    CFG['model_name'] = CFG['model_name'].split('|')
    
    CFG['seed'] = 20
    
    CFG, RESULT_PATH, var_type_dict = preprocess_runner(CFG)
    print(CFG)
    print(RESULT_PATH)
    print(var_type_dict)
    analysis_runner(CFG, var_type_dict, RESULT_PATH)
    
    return render(request, 'preprocess.html')
