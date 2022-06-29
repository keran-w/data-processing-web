"""
Django settings for app project.

Generated by 'django-admin startproject' using Django 2.2.5.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '(8*=6f42)mzmf!$+)olxgnk3*h^4(n@m=jj$3*q2a)i!nbh2$#'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.db.backends.sqlite3',
    'crispy_forms',
    'crispy_bootstrap5',
    'app'
]

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"

CRISPY_TEMPLATE_PACK = "bootstrap5"

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'app.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'app.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, "static"),
)

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')


# radio selections for variable types
VAR_TYPES = 'Variable Types', {
    'delete': '删除',
    'binary': '二值变量',
    'quan': '多值有序',
    'mult_order': '多值无序',
    'mult_disorder': '连续变量',
}

# checkbox selections for methods
SAMPLING_METHODS = 'Sampling Methods', {
    'SMO': 'SMO',
    'SSMO': 'SSMO',
    'BSMO': 'BSMO',
    'ADA': 'ADA',
    'ROS': 'ROS',
    'SMN': 'SMN',
}

IMPUTE_METHODS = 'Impute Methods', {
    'Simple': 'Simple',
    'KNN': 'KNN',
    'ISVD': 'ISVD',
    'Imput': 'Imput',
    'rf': 'rf',
    'optimal': 'optimal',
}

SELE_METHODS = 'Selection Methods', {
    'Las': 'Las',
    'RCV': 'RCV',
    'ENC': 'ENC',
    'Cat': 'Cat',
    'SVC': 'SVC',
    'RF': 'RF',
    'Ada': 'Ada',
    'GBC': 'GBC',
    'ExT': 'ExT',
    'BNB': 'BNB',
    'XGB': 'XGB',
    'LGBM': 'LGBM',
}

TRAIN_METHODS = 'Train Methods', {
    'Random Forest': 'Random Forest',
    'AdaBoost': 'AdaBoost',
    'Gradient Boosting': 'Gradient Boosting',
    'Multinomial Naive Bayes': 'Multinomial Naive Bayes',
    'Bernoulli Naive Bayes': 'Bernoulli Naive Bayes',
    'XGBoost': 'XGBoost',
    'LGBoost': 'LGBoost',
    'Gaussian Naive Bayes': 'Gaussian Naive Bayes',
    'Complement Naive Bayes': 'Complement Naive Bayes',
    'KNN': 'KNN',
    'CatBoost': 'CatBoost',
    'Decision Tree': 'Decision Tree',
    'QDA': 'QDA',
    'Extra Tree': 'Extra Tree',
    'SVC': 'SVC',
    'Passive Aggressive': 'Passive Aggressive',
    'LDA': 'LDA',
    'Logistic Regression': 'Logistic Regression',
    'SGD': 'SGD',
    'Bagging': 'Bagging',
    'MLP': 'MLP',
}
