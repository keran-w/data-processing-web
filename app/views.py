from multiprocessing import context
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import UploadFileForm
from .storage import OverwriteStorage
from .settings import MEDIA_ROOT
import os

def index(request):
    print('method', request.method)
    print('FILES', request.FILES)
    context = {}
    context['sections'] = 1
    context['display_table'] = False
    
    if request.method == 'POST' and request.FILES['file_data']:
        data = request.FILES['file_data']
        file_path = os.path.join(MEDIA_ROOT, data.name)
        if os.path.exists(file_path):
            os.remove(os.path.join(MEDIA_ROOT, data.name))
            
        fs = FileSystemStorage()
        file_name = fs.save(data.name, data)
        
        data_df = pd.read_excel(file_path).head(20).fillna('')
        context['file_name'] = file_name
        context['file_cols'] = list(data_df.columns)
        context['file_data'] = data_df.values.tolist()
        context['display_table'] = True
        
        return render(request, 'index.html', context)
        
    return render(request, 'index.html', context)
