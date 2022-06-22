from django.shortcuts import render

def index(request):
    context = {}
    context['name'] = "Hello World!"
    return render(request, 'index.html', context)