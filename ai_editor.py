"""AI编辑器助手模块

本模块实现了一个基于大语言模型的编辑器助手系统，通过整合多个管理器组件，提供智能文本生成和编辑功能。

主要功能：
- 基于RAG（检索增强生成）的文本生成
- 用户偏好学习和应用
- 知识库集成
- 编辑历史记录

核心类：
AIEditorAssistant - 编辑器助手的主类，整合了以下功能：
- 文本生成：使用LLM结合上下文进行智能生成
- 用户偏好管理：学习和应用用户的编辑偏好
- 向量存储：管理文档的向量表示
- 知识库访问：集成领域知识支持
- 数据库操作：记录编辑历史
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from preference_manager import UserPreferenceManager
from vector_manager import VectorStoreManager
from kb_manager import KnowledgeBaseManager
from db_manager import DatabaseManager

class AIEditorAssistant:
    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.preference_manager = UserPreferenceManager(llm)
        self.vector_manager = VectorStoreManager(embeddings)
        self.kb_manager = KnowledgeBaseManager(embeddings)
        self.db_manager = DatabaseManager(embeddings)

    def generate_text(self, user_text: str, prompt: str = "", top_k: int = 3) -> str:
        """使用LCEL实现的RAG方法生成文本内容
        
        Args:
            user_text: 用户输入的文本
            prompt: 额外的提示词
            top_k: 检索的相关文档数量
            
        Returns:
            生成的文本内容
        """
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.prompts import PromptTemplate
        from operator import itemgetter
        from typing import Sequence, Dict, Any
        
        # 1. 定义检索链
        def retrieve_docs(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            docs = self.vector_manager.search_similar_documents(input_dict["user_text"], k=top_k)
            return {"docs": [doc for doc in docs if doc.relevance_score > 0.7]}
            
        retriever_chain = RunnablePassthrough().assign(docs=retrieve_docs)
        
        # 2. 定义上下文增强链
        def enhance_context(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            preferences = self.preference_manager.get_preferences()
            kb_context = self.kb_manager.get_relevant_context(input_dict["user_text"])
            context_parts = [
                f"相关文档 {i+1}: {doc.content}"
                for i, doc in enumerate(input_dict["docs"])
            ]
            return {
                "context": "\n".join(context_parts),
                "preferences": preferences,
                "kb_context": kb_context
            }

        context_chain = RunnablePassthrough().assign(**enhance_context)
        
        # 3. 定义提示词模板
        prompt_template = PromptTemplate.from_template("""
{prompt}

用户输入:
{user_text}

相关上下文:
{context}

知识库参考:
{kb_context}

用户偏好:
{preferences}

请根据以上信息生成回复。注意保持与用户偏好一致，并充分利用相关上下文。
""")
        
        # 4. 组合完整的处理链
        chain = (
            {"user_text": itemgetter("user_text"), "prompt": itemgetter("prompt")}
            | retriever_chain
            | context_chain
            | prompt_template
            | self.llm
        )
        
        # 5. 执行链并返回结果
        return chain.invoke({"user_text": user_text, "prompt": prompt})


    def record_user_edit(self, original_text: str, edited_text: str, text_type: Optional[str] = None) -> None:
        """记录用户编辑"""
        # 更新用户偏好
        self.preference_manager.analyze_edits(original_text, edited_text, text_type)
        
        # 保存到向量存储
        self.vector_manager.add_content(edited_text, text_type)
        
        # 记录到数据库
        self.db_manager.record_edit(original_text, edited_text, text_type)