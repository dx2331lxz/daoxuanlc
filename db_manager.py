"""数据库管理模块

本模块负责管理项目中的数据库操作，包括用户偏好的存储和检索，以及向量数据的管理。主要功能包括：

1. 用户偏好管理：
   - 保存用户对不同类型文本的编辑偏好
   - 检索和获取用户的历史偏好记录
   - 支持按文本类型分类存储偏好

2. 向量数据管理：
   - 通过VectorStoreManager管理文档的向量存储
   - 支持相似文档的检索功能
   - 提供专业领域知识的存储和检索接口

与其他模块的交互：
- 与editor.models模块交互，使用UserPreference模型进行数据持久化
- 与vector_manager模块集成，处理文档的向量化存储和检索
- 为preference_manager模块提供数据存储支持
"""

from django.db import connection
from editor.models import UserPreference
from typing import List, Dict, Optional
import json
from vector_manager import VectorStoreManager

class DatabaseManager:
    def __init__(self, embeddings):
        self.vector_manager = VectorStoreManager(embeddings)
        self.connection = connection
    
    def save_preference(self, user_id: str, text_type: str, preference_key: str, preference_value: str):
        """保存用户偏好"""
        try:
            preference = UserPreference(
                user_id=user_id,
                text_type=text_type,
                preference_key=preference_key,
                preference_value=preference_value
            )
            preference.save()
            return True
        except Exception as e:
            print(f"保存用户偏好时出错: {e}")
            return False
    
    def get_preferences(self, user_id: str, text_type: str = "general") -> List[str]:
        """获取指定类型的用户偏好"""
        try:
            preferences = UserPreference.objects.filter(
                user_id=user_id,
                text_type=text_type
            ).values_list('preference_value', flat=True)
            return list(preferences)
        except Exception as e:
            print(f"获取用户偏好时出错: {e}")
            return []
    
    def analyze_edits(self, user_id: str, original_text: str, edited_text: str, text_type: str = "general", preference_key: str = "edit_preference"):
        """分析用户编辑，提取偏好"""
        if original_text == edited_text:
            return  # 没有修改，不需要分析
        
        # 保存编辑历史作为偏好
        edit_data = {
            "original": original_text,
            "edited": edited_text
        }
        
        self.save_preference(
            user_id=user_id,
            text_type=text_type,
            preference_key=preference_key,
            preference_value=json.dumps(edit_data, ensure_ascii=False)
        )
    
    def save_knowledge(self, text_type: str, documents: List) -> bool:
        """保存专业领域知识到向量存储"""
        return self.vector_manager.save_vector_store(text_type, documents)
    
    def search_knowledge(self, text_type: str, query: str, k: int = 5) -> List[Dict]:
        """搜索相关的专业领域知识"""
        return self.vector_manager.search_similar_documents(text_type, query, k)
