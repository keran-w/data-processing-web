import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseNotFound, HttpResponse
from django.core.files.storage import FileSystemStorage

from .forms import ConfigForm
from .storage import OverwriteStorage
from .settings import MEDIA_ROOT
# from .core.process import DataProcesser

NAN = -999.99999999
global FILEPATH


def index(request):

    context = {}
    context['sections'] = 1
    context['display_table'] = False

    if request.method == 'POST' and request.FILES['file_data'] and 'show' in request.POST:
        data = request.FILES['file_data']
        file_path = os.path.join(MEDIA_ROOT, data.name)
        global FILEPATH
        FILEPATH = file_path
        if os.path.exists(file_path):
            os.remove(os.path.join(MEDIA_ROOT, data.name))

        fs = FileSystemStorage()
        file_name = fs.save(data.name, data)

        try:
            data_df = pd.read_excel(file_path, nrows=30)
        except:
            data_df = pd.read_csv(file_path, nrows=30)
        data_df = data_df.fillna(NAN).round(3)
        data_df = data_df.replace(NAN, '')
        context['file_name'] = file_name
        context['file_cols'] = list(data_df.columns)
        context['file_data'] = data_df.values.tolist()
        context['display_table'] = True

        return render(request, 'index.html', context)

    return render(request, 'index.html', context)


def config(request):

    try:
        FILEPATH
    except:
        return HttpResponseNotFound(404)

    try:
        data_df = pd.read_excel(FILEPATH)
    except:
        data_df = pd.read_csv(FILEPATH)

    form_config = dict(
        variables=list(data_df.columns),
        data_df=data_df
    )
    
    if request.method == 'POST':
        form = ConfigForm(request.POST, **form_config)
        if form.is_valid():
            form_data = form.cleaned_data
            print(form_data)
            return HttpResponse(FILEPATH + '<br>' + str(form_data) + '<br>' + str(form_config))
    else:
        form = ConfigForm(**form_config)

    return render(request, 'config.html', {'form': form})
