from pathlib import Path
from env_loader import get_env_variable

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

# 最小化Django配置，仅用于ORM
SECRET_KEY = get_env_variable('DJANGO_SECRET_KEY')
DEBUG = get_env_variable('DJANGO_DEBUG', 'False') == 'True'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': get_env_variable('DB_NAME'),
        'USER': get_env_variable('DB_USER'),
        'PASSWORD': get_env_variable('DB_PASSWORD'),
        'HOST': get_env_variable('DB_HOST'),
        'PORT': get_env_variable('DB_PORT'),
    }
}

# 仅包含必要的应用
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'editor.apps.EditorConfig',
]
