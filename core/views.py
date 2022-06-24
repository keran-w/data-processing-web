from calendar import c
from multiprocessing import context
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage


def index(request):
    print(request)

    if request.method == 'POST':
        # and request.FILES['data']
        data = request.FILES['data']
        print()
        print()
        print(data)
        print()
        print()
        fs = FileSystemStorage()
        filename = fs.save(data.name, data)
        data_html = pd.read_excel(filename).head(20).to_html()
        return render(request, 'index.html', {
            data: data_html,
        })

    return render(request, 'index.html')
