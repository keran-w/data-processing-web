from multiprocessing import context
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .storage import OverwriteStorage
from .settings import MEDIA_ROOT
import os
from django.http import HttpResponseNotFound

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
    context = {}
    
    try:
        FILEPATH
    except:
        return HttpResponseNotFound(404)
    
    try:
        data_df = pd.read_excel(FILEPATH, nrows=1)
    except:
        data_df = pd.read_csv(FILEPATH, nrows=1)
    
    context['file_cols'] = list(data_df.columns)
    context['items'] = ['AAAA', 'BBBBBB', 'C', 'DDD']
    # context['items'] = [[f'{i}', j] for i, j in enumerate(context['items'])]
    
    return render(request, 'config.html', context)
