# Data Processing Web

### Using Virtual Environment

python -m venv .venv

conda create -n env-data-proc-web python=3.9

.venv/Scripts/Activate.ps1

pip install -r requirements.txt

### Using pip

pip install -r requirements.txt --user

### Setup

python manage.py migrate

### Run

python manage.py runserver 8080
