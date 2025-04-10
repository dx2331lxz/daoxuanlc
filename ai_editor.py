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

from this import d
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from preference_manager import UserPreferenceManager
from vector_manager import VectorStoreManager
from kb_manager import KnowledgeBaseManager
from db_manager import DatabaseManager
from classifiers import TextTypeClassifier

class AIEditorAssistant:
    def __init__(self, llm: BaseChatModel,streaming_llm: BaseChatModel ,embeddings: Embeddings):
        self.llm = llm
        self.streaming_llm = streaming_llm
        self.embeddings = embeddings
        self.preference_manager = UserPreferenceManager(llm)
        self.vector_manager = VectorStoreManager(embeddings)
        self.kb_manager = KnowledgeBaseManager(embeddings)
        self.db_manager = DatabaseManager(embeddings)  # 确保传递了embeddings参数
        self.text_classifier = TextTypeClassifier(llm)

    async def generate_text(self, user_text: str, prompt: str = "", top_k: int = 3, stream: bool = False) -> str:
        """使用LCEL实现的RAG方法生成文本内容"""
            
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.prompts import PromptTemplate
        from operator import itemgetter
        from typing import Dict, Any
        
        # 1. 定义检索链
        def retrieve_docs(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            # 根据用户输入和提示词确定文本类型
            query = input_dict['user_text'] + '\n\n' + input_dict['prompt'] if input_dict['user_text'] else input_dict['prompt']
            text_type = self.text_classifier.classify(query)
            print(f"文本类型: {text_type}")
            docs = self.vector_manager.search_similar_documents(
                query=query,
                k=top_k,
                text_type=text_type
            )
            print(f"检索到的文档: {docs}")
            return {"docs": [doc for doc in docs if doc.get("score", 0) > 0.7]}
            
        retriever_chain = (
            RunnablePassthrough()
            | {
                "docs": lambda x: retrieve_docs(x)["docs"],
                "user_text": itemgetter("user_text"),
                "prompt": itemgetter("prompt")
            }
        )
        
        # 2. 定义上下文增强链
        def enhance_context(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            preferences = self.preference_manager.get_preferences()
            context_parts = [
                f"相关文档 {i+1}: {doc['content']}"
                for i, doc in enumerate(input_dict["docs"])
            ]
            return {
                "context": "\n".join(context_parts),
                "preferences": preferences
            }

        context_chain = (
            RunnablePassthrough()
            | {
                "context": lambda x: enhance_context(x)["context"],
                "preferences": lambda x: enhance_context(x)["preferences"],
                "user_text": itemgetter("user_text"),
                "prompt": itemgetter("prompt")
            }
        )
        
        # 3. 定义提示词模板
        prompt_template = PromptTemplate.from_template("""
{prompt}

用户输入:
{user_text}

相关上下文:
{context}

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
            | (self.streaming_llm if stream else self.llm)
        )     
        
        # 5. 执行链并返回结果
        if stream:
            async for chunk in chain.astream({"user_text": user_text, "prompt": prompt}):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                elif hasattr(chunk, 'text'):
                    yield chunk.text
                else:
                    text = str(chunk)
                    if text.strip():
                        yield text
        else:
            result = await chain.ainvoke({"user_text": user_text, "prompt": prompt})
            if hasattr(result, 'content'):
                yield result.content
            elif hasattr(result, 'text'):
                yield result.text
            else:
                yield str(result)

    async def generate_text_with_temp_context(self, user_text: str, prompt: str, temp_vectorstore=None, top_k: int = 6, stream: bool = False):
        """使用临时向量存储生成文本内容"""
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.prompts import PromptTemplate
        from operator import itemgetter
        from typing import Dict, Any
        
        # 1. 定义检索链，使用临时向量存储
        def retrieve_docs(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            query = input_dict['user_text'] + '\n\n' + input_dict['prompt'] if input_dict['user_text'] else input_dict['prompt']
            if temp_vectorstore:
                docs = temp_vectorstore.similarity_search_with_score(query, k=top_k)

                return {"docs": [{
                    "content": doc[0].page_content,
                    "score": doc[1]
                } for doc in docs if doc[1] > 0.7]}
            return {"docs": []}
            
        retriever_chain = (
            RunnablePassthrough()
            | {
                "docs": lambda x: retrieve_docs(x)["docs"],
                "user_text": itemgetter("user_text"),
                "prompt": itemgetter("prompt")
            }
        )
        
        # 2. 定义上下文增强链
        def enhance_context(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            preferences = self.preference_manager.get_preferences()
            kb_context = self.kb_manager.get_relevant_context(input_dict["user_text"])
            context_parts = [
                f"相关文档 {i+1}: {doc['content']}"
                for i, doc in enumerate(input_dict["docs"])
            ]
            return {
                "context": "\n".join(context_parts),
                "preferences": preferences,
                "kb_context": kb_context
            }

        context_chain = (
            RunnablePassthrough()
            | {
                "context": lambda x: enhance_context(x)["context"],
                "preferences": lambda x: enhance_context(x)["preferences"],
                "user_text": itemgetter("user_text"),
                "prompt": itemgetter("prompt")
            }
        )
        
        # 3. 定义提示词模板
        prompt_template = PromptTemplate.from_template("""
{prompt}

用户输入:
{user_text}

相关上下文:
{context}

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
            | self.streaming_llm
        )
        
        # 5. 执行链并返回结果
        if stream:
            async for chunk in chain.astream({"user_text": user_text, "prompt": prompt}):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                elif hasattr(chunk, 'text'):
                    yield chunk.text
                else:
                    text = str(chunk)
                    if text.strip():
                        yield text
        else:
            result = await chain.ainvoke({"user_text": user_text, "prompt": prompt})
            if hasattr(result, 'content'):
                yield result.content
            elif hasattr(result, 'text'):
                yield result.text
            else:
                yield str(result)

    def record_user_edit(self, original_text: str, edited_text: str, user_id: str = None) -> None:
        """记录用户编辑"""
        # 更新用户偏好，使用默认的general类型
        self.preference_manager.analyze_edits(original_text, edited_text)