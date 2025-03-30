from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model="Pro/BAAI/bge-m3",  # 硅基流动的嵌入模型名称
    openai_api_base="https://api.siliconflow.cn/v1",  # 硅基流动API端点
    openai_api_key=""  # 替换为硅基流动API密钥
)

# 初始化LLM
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",  # DeepSeek的API端点
    openai_api_key="",  # 替换为您的DeepSeek API密钥
    temperature=0.7  # 可选参数，控制生成结果的随机性
)
