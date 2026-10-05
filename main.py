import os
from functools import lru_cache

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv
from agent import Agent

app = FastAPI()
load_dotenv()

@lru_cache(maxsize=1)
def get_agent() -> Agent:
    """Create the DeepSeek agent only when the chat endpoint is used."""

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")

    return Agent(
        api_key,
        "https://api.deepseek.com",
        "deepseek-flash"
    )

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        reply = get_agent().run(request.message)

        return {
            "reply": reply
        }

    except Exception as exc:
        import traceback

        print("\n========== /chat 真实异常 ==========")
        traceback.print_exc()
        print("====================================\n")

        raise HTTPException(
            status_code=502,
            detail=str(exc)
        ) from exc

html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>我的 AI</title>

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }

        .container {
            width: 800px;
            margin: 40px auto;
            background: white;
            border-radius: 12px;
            overflow: hidden;
        }

        .header {
            padding: 20px;
            font-size: 22px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }

        .messages {
            height: 500px;
            padding: 20px;
            overflow-y: auto;
        }

        .message {
            margin-bottom: 15px;
        }

        .user {
            text-align: right;
        }

        .input-area {
            display: flex;
            padding: 15px;
            border-top: 1px solid #ddd;
        }

        input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 8px;
        }

        button {
            margin-left: 10px;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        我的 AI
    </div>

    <div class="messages" id="messages">

        <div class="message">
            AI：你好，我是你的 AI Agent。
        </div>

    </div>

    <div class="input-area">

        <input
            id="input"
            type="text"
            placeholder="输入消息..."
        >

        <button onclick="sendMessage()">
            发送
        </button>

    </div>

</div>


<script>

const messages = document.getElementById("messages");

async function sendMessage() {

    const input = document.getElementById("input");

    const message = input.value;

    if (!message) {
        return;
    }

    addMessage(message, "user");

    input.value = "";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const text = await response.text();

        let data;

        try {
            data = JSON.parse(text);
        } catch (parseError) {
            throw new Error(
                `服务器返回了非 JSON 内容：${text}`
            );
        }

        if (!response.ok) {
            throw new Error(
                data.detail || "请求失败"
            );
        }

        addMessage(
            `AI：${data.reply ?? "暂时没有回复"}`
        );

    } catch (error) {

        addMessage(
            `错误：${error.message}`
        );

    }
}
function addMessage(text, className = "") {
    const element = document.createElement("div");
    element.className = `message ${className}`.trim();
    element.textContent = text;
    messages.appendChild(element);
    messages.scrollTop = messages.scrollHeight;
}

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return html
