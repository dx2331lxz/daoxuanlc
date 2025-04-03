"""知识库管理模块

本模块实现了一个基于向量存储的知识库管理系统，用于存储和检索不同类型的专业领域知识。

主要功能：
- 自动加载和管理多个领域的知识库文档
- 文档向量化存储和相似度检索
- 支持学术、技术、创意和商业等多种文本类型

核心类：
KnowledgeBaseManager - 知识库管理器，提供以下功能：
- 知识库加载：自动加载不同领域的文档
- 文档分割：使用递归字符分割器处理长文本
- 向量存储：使用FAISS进行高效的向量存储和检索
- 相似度搜索：基于文本嵌入的相似度检索
"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 知识库目录
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_bases")

# 文本类型及其对应的知识库目录
TEXT_TYPES = {
    "academic": os.path.join(KNOWLEDGE_BASE_DIR, "academic"),
    "technical": os.path.join(KNOWLEDGE_BASE_DIR, "technical"),
    "creative": os.path.join(KNOWLEDGE_BASE_DIR, "creative"),
    "business": os.path.join(KNOWLEDGE_BASE_DIR, "business"),
}



class KnowledgeBaseManager:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vector_stores = {}
        self._load_knowledge_bases()
    
    def _load_knowledge_bases(self):
        """加载所有知识库"""
        for text_type, directory in TEXT_TYPES.items():
            try:
                # 检查目录是否存在
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    continue
                # 加载目录中的文本文件
                loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
                documents = loader.load()
                
                # 如果没有文档，尝试加载README.md
                if not documents:
                    readme_path = os.path.join(directory, "README.md")
                    if os.path.exists(readme_path):
                        loader = TextLoader(readme_path)
                        documents = loader.load()
                # 如果有文档，创建向量存储
                if documents:
                    split_docs = text_splitter.split_documents(documents)
                    self.vector_stores[text_type] = FAISS.from_documents(split_docs, self.embeddings)
            except Exception as e:
                print(f"加载知识库 {text_type} 时出错: {e}")
    
    def get_retriever(self, text_type: str):
        """获取指定类型的检索器"""
        if text_type in self.vector_stores:
            return self.vector_stores[text_type].as_retriever()
        return None
        
    def get_relevant_context(self, query_text: str, top_k: int = 3) -> str:
        """从所有知识库中检索与查询文本相关的内容
        
        Args:
            query_text: 查询文本
            top_k: 每个知识库返回的文档数量
            
        Returns:
            组合后的相关上下文字符串
        """
        if not query_text or not isinstance(query_text, str):
            return ""
            
        all_contexts = []
        
        # 从每个知识库中检索相关文档
        for text_type, vector_store in self.vector_stores.items():
            try:
                docs = vector_store.similarity_search_with_score(query_text, k=top_k)
                if docs:
                    # 只保留相关度较高的文档（分数大于0.7）
                    relevant_docs = [doc for doc, score in docs if score > 0.7]
                    if relevant_docs:
                        context = f"\n{text_type}知识库相关内容:\n" + "\n".join(
                            f"- {doc.page_content}" 
                            for doc in relevant_docs
                        )
                        all_contexts.append(context)
            except Exception as e:
                print(f"从{text_type}知识库检索内容时出错: {e}")
                continue
        
        # 如果没有找到相关内容，返回空字符串
        if not all_contexts:
            return ""
            
        # 组合所有知识库的相关内容
        return "\n".join(all_contexts)