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