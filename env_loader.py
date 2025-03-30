import os
from pathlib import Path
from dotenv import load_dotenv

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载环境变量
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(env_path)

# 获取环境变量的辅助函数
def get_env_variable(var_name, default=None):
    """获取环境变量，如果不存在则返回默认值"""
    return os.environ.get(var_name, default)