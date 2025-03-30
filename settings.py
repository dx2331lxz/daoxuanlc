from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

# 最小化Django配置，仅用于ORM
SECRET_KEY = 'django-insecure-your-secret-key-here'
DEBUG = False

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'editor',
        'USER': 'editor',
        'PASSWORD': '',
        'HOST': '',
        'PORT': '3306',
    }
}

# 仅包含必要的应用
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'editor.apps.EditorConfig',
]
