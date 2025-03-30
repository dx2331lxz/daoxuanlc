"""文本分类器模块

本模块实现了一个基于大语言模型的文本类型分类系统，用于自动识别和分类不同类型的文本内容。

主要功能：
- 文本类型自动识别
- 支持多种文本类型（学术、技术、创意、商业）
- 基于LLM的智能分类

核心类：
TextTypeClassifier - 文本类型分类器，提供以下功能：
- 文本分析：使用LLM分析文本特征
- 类型判断：将文本分类为预定义的类型
- 智能默认：对未匹配类型提供默认处理
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict

# 文本类型及其对应的知识库目录
TEXT_TYPES = {
    "academic": "knowledge_bases/academic",
    "technical": "knowledge_bases/technical",
    "creative": "knowledge_bases/creative",
    "business": "knowledge_bases/business",
}

class TextTypeClassifier:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
        你是一个文本类型分类器。请分析以下文本内容，并确定它属于哪种类型。
        可能的类型有：academic（学术论文）、technical（技术文档）、creative（创意写作）、business（商业文档）。
        只返回一个类型名称，不要有任何其他解释。
        
        文本内容：
        {text}
        """)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def classify(self, text: str) -> str:
        """识别文本类型"""
        result = self.chain.invoke({"text": text})
        # 确保结果是有效的文本类型
        result = result.strip().lower()
        if result not in TEXT_TYPES:
            # 默认为创意写作
            return "creative"
        return result