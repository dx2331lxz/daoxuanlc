"""日志管理模块

本模块提供了统一的日志记录功能，支持请求日志和错误日志的记录。
主要特性：
- 按日期自动分类日志文件
- 详细的错误堆栈信息记录
- 不同级别的日志支持
- 自定义日志格式
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
import traceback
from datetime import datetime

class LogManager:
    def __init__(self):
        self.logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        self._ensure_log_dir()
        
        # 配置请求日志记录器
        self.request_logger = self._setup_logger('request', 'requests.log',
            '%(asctime)s - %(levelname)s - %(message)s')
            
        # 配置错误日志记录器
        self.error_logger = self._setup_logger('error', 'errors.log',
            '%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    
    def _ensure_log_dir(self):
        """确保日志目录存在"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
    def _setup_logger(self, name: str, filename: str, format_str: str) -> logging.Logger:
        """设置日志记录器
        
        Args:
            name: 日志记录器名称
            filename: 日志文件名
            format_str: 日志格式字符串
            
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 创建按天轮转的文件处理器
        handler = TimedRotatingFileHandler(
            os.path.join(self.logs_dir, filename),
            when='midnight',
            interval=1,
            backupCount=30,  # 保留30天的日志
            encoding='utf-8'
        )
        
        # 设置日志格式
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(handler)
        
        return logger
    
    def log_request(self, message: str, level: str = 'info') -> None:
        """记录请求日志
        
        Args:
            message: 日志消息
            level: 日志级别，默认为'info'
        """
        log_method = getattr(self.request_logger, level.lower(), self.request_logger.info)
        log_method(message)
    
    def log_error(self, error: Exception, module: str = None) -> None:
        """记录错误日志
        
        Args:
            error: 异常对象
            module: 发生错误的模块名称
        """
        error_info = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        module_info = f'[{module}] ' if module else ''
        self.error_logger.error(f'{module_info}Error occurred:\n{error_info}')

    def auto_log_request(self, func):
        """自动记录请求的装饰器"""
        def wrapper(*args, **kwargs):
            try:
                # 自动记录请求开始
                self.log_request(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
                result = func(*args, **kwargs)
                # 自动记录请求成功
                self.log_request(f"{func.__name__} executed successfully")
                return result
            except Exception as e:
                # 自动记录错误
                self.log_error(e, module=func.__module__)
                raise
        return wrapper

# 使用示例:
# @log_manager.auto_log_request
# def your_function(param1, param2):
#     ...

# 创建全局日志管理器实例
log_manager = LogManager()