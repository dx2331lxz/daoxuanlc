"""向量存储管理模块

本模块实现了一个基于FAISS的向量存储管理系统，用于高效存储和检索文本的向量表示。

主要功能：
- 文本向量化存储和管理
- 本地持久化存储
- 高效的相似度检索

核心类：
VectorStoreManager - 向量存储管理器，提供以下功能：
- 向量存储：将文档转换为向量并存储
- 本地持久化：支持向量数据的本地保存和加载
- 相似度检索：基于向量相似度的文档检索
- 内存管理：高效的向量存储缓存机制
"""

from langchain_community.vectorstores import FAISS
from typing import List, Dict, Optional
import os

class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_stores = {}
        self.knowledge_base_dir = 'knowledge_bases'
        self.vector_store_dir = 'vector_stores'
        
        # 确保向量存储目录存在
        if not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)
        
        # 加载知识库文件到向量存储
        self._load_knowledge_bases()
    
    def _load_knowledge_bases(self):
        """加载知识库文件到向量存储"""
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import CharacterTextSplitter
        
        # 遍历知识库目录
        for domain in ['academic', 'technical', 'creative', 'business']:
            domain_path = os.path.join(self.knowledge_base_dir, domain)
            if not os.path.exists(domain_path):
                continue
                
            documents = []
            # 遍历目录下的所有文件
            for root, _, files in os.walk(domain_path):
                for file in files:
                    if file.endswith('.md') or file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        try:
                            # 加载文档
                            loader = TextLoader(file_path)
                            docs = loader.load()
                            
                            # 分割文档
                            text_splitter = CharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200
                            )
                            split_docs = text_splitter.split_documents(docs)
                            documents.extend(split_docs)
                        except Exception as e:
                            print(f"加载文件 {file_path} 时出错: {e}")
            
            # 保存到向量存储
            if documents:
                self.save_vector_store(domain, documents)
    
    def save_vector_store(self, text_type: str, documents: List):
        """保存文档到向量存储"""
        try:
            # 使用FAISS创建向量存储
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # 保存向量存储到本地
            save_path = os.path.join(self.vector_store_dir, f'{text_type}')
            vector_store.save_local(save_path)
            
            # 更新内存中的向量存储
            self.vector_stores[text_type] = vector_store
            return True
        except Exception as e:
            print(f"保存向量存储时出错: {e}")
            return False
    
    def load_vector_store(self, text_type: str) -> Optional[FAISS]:
        """加载向量存储"""
        try:
            # 如果内存中已有该类型的向量存储，直接返回
            if text_type in self.vector_stores:
                return self.vector_stores[text_type]
            
            # 从本地加载向量存储
            load_path = os.path.join(self.vector_store_dir, f'{text_type}')
            if os.path.exists(load_path):
                vector_store = FAISS.load_local(load_path, self.embeddings)
                self.vector_stores[text_type] = vector_store
                return vector_store
            return None
        except Exception as e:
            print(f"加载向量存储时出错: {e}")
            return None


    def search_similar_documents(self, text_type: str, query: str, k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        # 输入验证
        if not query or not isinstance(query, str):
            print("错误：查询文本不能为空且必须是字符串类型")
            return []
            
        try:
            vector_store = self.load_vector_store(text_type)
            if not vector_store:
                return []
            
            # 执行相似度搜索
            results = vector_store.similarity_search_with_score(query, k=k)
            
            # 格式化结果
            formatted_results = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                } for doc, score in results
            ]
            
            return formatted_results
        except Exception as e:
            print(f"搜索相似文档时出错: {str(e)}")
            return []
