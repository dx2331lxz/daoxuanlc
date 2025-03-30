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