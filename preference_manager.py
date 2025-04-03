from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
# 延迟导入以避免循环依赖
from django.apps import apps

def get_mysql_preference_manager():
    from db_manager import DatabaseManager
    from config import embeddings
    return DatabaseManager(embeddings)

class UserPreferenceManager:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.db_manager = None
        
    def _ensure_db_manager(self):
        if self.db_manager is None:
            self.db_manager = get_mysql_preference_manager()
        self.summarize_prompt = ChatPromptTemplate.from_template("""
        分析以下用户对AI生成内容的修改，总结出用户的偏好。
        
        原始AI生成内容：
        {original_text}
        
        用户修改后的内容：
        {edited_text}
        
        请总结用户的偏好（例如：风格、格式、内容深度、专业术语使用等方面）：
        """)
        self.summarize_chain = self.summarize_prompt | self.llm | StrOutputParser()
    
    def analyze_edits(self, original_text: str, edited_text: str, text_type: str = "general"):
        """分析用户编辑，提取偏好"""
        self._ensure_db_manager()
        if original_text == edited_text:
            return  # 没有修改，不需要分析
        
        # 使用LLM总结用户偏好
        preference = self.summarize_chain.invoke({
            "original_text": original_text,
            "edited_text": edited_text
        })
        
        # 保存偏好到数据库
        self.db_manager.save_preference(
            user_id="default",  # 暂时使用默认用户ID
            text_type=text_type,
            preference_key="llm_analysis",
            preference_value=preference
        )
        
        # 同时保存到通用偏好
        if text_type != "general":
            self.db_manager.save_preference(
                user_id="default",
                text_type="general",
                preference_key="llm_analysis",
                preference_value=preference
            )
    
    def get_preferences(self, text_type: str = "general") -> List[str]:
        """获取指定类型的用户偏好"""
        self._ensure_db_manager()
        return self.db_manager.get_preferences("default", text_type)