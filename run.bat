Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
.venv\Scripts\Activate.ps1
start http://127.0.0.1:8081/ && python .\manage.py runserver 8081