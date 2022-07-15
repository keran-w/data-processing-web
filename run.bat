Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
call .venv\Scripts\Activate.bat
start http://127.0.0.1:8082/ && python .\manage.py runserver 8082