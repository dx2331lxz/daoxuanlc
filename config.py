from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from env_loader import get_env_variable

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model="Pro/BAAI/bge-m3",  # 硅基流动的嵌入模型名称
    openai_api_base=get_env_variable('SILICONFLOW_API_BASE'),  # 硅基流动API端点
    openai_api_key=get_env_variable('SILICONFLOW_API_KEY')  # 从环境变量获取API密钥
)

# 初始化LLM
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base=get_env_variable('DEEPSEEK_API_BASE'),  # DeepSeek的API端点
    openai_api_key=get_env_variable('DEEPSEEK_API_KEY'),  # 从环境变量获取API密钥
    temperature=0.7  # 可选参数，控制生成结果的随机性
)
