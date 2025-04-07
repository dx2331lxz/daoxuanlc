from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from env_loader import get_env_variable
from langchain.callbacks.base import BaseCallbackHandler

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model="Pro/BAAI/bge-m3",  # 硅基流动的嵌入模型名称
    openai_api_base=get_env_variable('SILICONFLOW_API_BASE'),  # 硅基流动API端点
    openai_api_key=get_env_variable('SILICONFLOW_API_KEY')  # 从环境变量获取API密钥
)

# 初始化非流式LLM
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base=get_env_variable('DEEPSEEK_API_BASE'),  # DeepSeek的API端点
    openai_api_key=get_env_variable('DEEPSEEK_API_KEY'),  # 从环境变量获取API密钥
    temperature=0.7  # 可选参数，控制生成结果的随机性
)

# # 初始化流式LLM
# streaming_llm = ChatOpenAI(
#     model_name="deepseek-chat",
#     openai_api_base=get_env_variable('DEEPSEEK_API_BASE'),
#     openai_api_key=get_env_variable('DEEPSEEK_API_KEY'),
#     streaming=True,  # 启用流式输出
#     callbacks=[StreamingStdOutCallbackHandler()],  # 添加流式输出回调处理器
#     temperature=0.7
# )

# 自定义不输出到终端的回调函数
class SilentCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 不进行任何操作，静默处理
        pass

# 初始化流式LLM
streaming_llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base=get_env_variable('DEEPSEEK_API_BASE'),
    openai_api_key=get_env_variable('DEEPSEEK_API_KEY'),
    streaming=True,  # 启用流式输出
    callbacks=[SilentCallbackHandler()],  # 使用自定义的静默回调处理器
    temperature=0.7
)
