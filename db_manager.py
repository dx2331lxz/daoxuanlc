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