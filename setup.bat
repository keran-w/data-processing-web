Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
python .\manage.py migrate