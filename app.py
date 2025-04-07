import os
import django
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, HttpUrl
from typing import Optional
from langserve import add_routes
from langchain_core.runnables import RunnablePassthrough
from config import llm, embeddings, streaming_llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
assistant = AIEditorAssistant(llm, streaming_llm, embeddings)

# 请求模型
class TextGenerationRequest(BaseModel):
    user_text: Optional[str] = None
    prompt: str
    file: Optional[UploadFile] = None
    url: Optional[HttpUrl] = None

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
        # 确保user_text不为None，如果为None则使用空字符串
        return assistant.generate_text(request.user_text or "", request.prompt)
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


@app.post("/generate-with-context")
async def generate_with_context(
    prompt: str = Form(...),
    user_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    url: Optional[HttpUrl] = Form(None)
):
    """生成带有文件或URL上下文的文本内容（流式输出）"""
    from fastapi.responses import StreamingResponse
    
    try:
        context = ""
        if file:
            content = await file.read()
            file_extension = file.filename.split('.')[-1].lower()
            
            if file_extension == 'txt':
                context += content.decode('utf-8')
            elif file_extension == 'md':
                context += content.decode('utf-8')
            elif file_extension == 'pdf':
                from PyPDF2 import PdfReader
                import io
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    context += page.extract_text()
            elif file_extension == 'docx' or file_extension == 'doc':
                from docx import Document
                import io
                doc = Document(io.BytesIO(content))
                for para in doc.paragraphs:
                    context += para.text + '\n'
            elif file_extension == 'ppt' or file_extension == 'pptx':
                from pptx import Presentation
                import io
                prs = Presentation(io.BytesIO(content))
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            context += shape.text + '\n'
            else:
                raise HTTPException(status_code=400, detail="不支持的文件类型")
        if url:
            # 使用requests获取URL内容
            import requests
            from bs4 import BeautifulSoup
            response = requests.get(str(url))
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            context += soup.get_text()
        
        if context:
            # 使用文本分割器将文档分成合适大小的块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
            )
            text_chunks = text_splitter.split_text(context)
            # 创建临时向量存储
            from langchain_community.vectorstores import FAISS
            temp_vectorstore = FAISS.from_texts(text_chunks, embeddings)
        else:
            temp_vectorstore = None
        
        
        # 使用异步生成器进行流式输出
        async def generate_stream():
            try:
                async for chunk in assistant.generate_text_with_temp_context(
                    user_text=user_text,
                    prompt=prompt,
                    temp_vectorstore=temp_vectorstore,
                    stream=True
                ):
                    # 将每个文本块格式化为SSE消息格式
                    yield f"data: {str(chunk)}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"  # 错误处理


        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Encoding": "none"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 创建另一个Runnable对象来包装generate_with_context方法
generate_with_context_runnable = (
    RunnablePassthrough()
    | {
        "user_text": lambda x: x.get("user_text", ""),
        "prompt": lambda x: x["prompt"],
        "file": lambda x: x.get("file", None),
        "url": lambda x: x.get("url", None),
        "result": lambda x: assistant.generate_text_with_temp_context(
            user_text=x.get("user_text", ""),
            prompt=x["prompt"],
            temp_vectorstore=None  # 需要根据file/url内容创建
        )
    }
)

# 添加LangServe路由
add_routes(
    app,
    generate_runnable,
    path="/langserve/generate",
    input_type=TextGenerationRequest,
    config_keys=["user_text", "prompt"]
)

add_routes(
    app,
    generate_with_context_runnable,
    path="/langserve/generate-with-context",
    input_type=TextGenerationRequest,
    config_keys=["user_text", "prompt", "file", "url"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    



    

