import os
import django
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langserve import add_routes
from langchain_core.runnables import RunnablePassthrough
from config import llm, embeddings

# 设置Django环境（仅用于ORM）
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
django.setup()

# 创建FastAPI应用
app = FastAPI(
    title="AI编辑器助手API",
    description="基于LangChain的AI编辑器助手API服务",
    version="1.0.0"
)

# 在Django初始化完成后导入和创建AI编辑器助手实例
from ai_editor import AIEditorAssistant
assistant = AIEditorAssistant(llm, embeddings)

# 请求模型
class TextGenerationRequest(BaseModel):
    user_text: Optional[str] = None
    prompt: str

# 创建一个Runnable对象来包装generate_text方法
generate_runnable = (
    RunnablePassthrough()
    | {
        "user_text": lambda x: x.get("user_text", ""),
        "prompt": lambda x: x["prompt"],
        "result": lambda x: assistant.generate_text(x.get("user_text", ""), x["prompt"])
    }
)

class UserEditRequest(BaseModel):
    original_text: str
    edited_text: str
    text_type: Optional[str] = None

# 添加API路由
@app.post("/generate", response_model=str)
async def generate_text(request: TextGenerationRequest):
    """生成文本内容"""
    try:
        return assistant.generate_text(request.user_text, request.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-edit")
async def record_edit(request: UserEditRequest):
    """记录用户编辑"""
    try:
        assistant.record_user_edit(
            request.original_text,
            request.edited_text,
            request.text_type
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preferences/{text_type}")
async def get_preferences(text_type: str = "general"):
    """获取用户偏好"""
    try:
        return {"preferences": assistant.preference_manager.get_preferences(text_type)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 添加LangServe路由
add_routes(
    app,
    generate_runnable,
    path="/langserve/generate",
    input_type=TextGenerationRequest,
    config_keys=["user_text", "prompt"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    

