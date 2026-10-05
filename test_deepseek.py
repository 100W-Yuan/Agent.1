from pydantic import BaseModel # type: ignore
from fastapi import FastAPI # type: ignore
from fastapi.responses import HTMLResponse # type: ignore
from openai import OpenAI # type: ignore


app = FastAPI()


# DeepSeek 客户端
client = OpenAI(
    api_key="sk-fc06b81600f948b9a8b64cd89d3743b6",
    base_url="https://api.deepseek.com"
)


class ChatRequest(BaseModel):
    message: str


# 聊天接口
@app.post("/chat")
def chat(request: ChatRequest):

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": request.message
            }
        ],
        max_tokens=1000
    )

    return {
        "reply": response.choices[0].message.content
    }


# 网页
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

async function sendMessage() {

    const input = document.getElementById("input");

    const message = input.value;

    if (!message) {
        return;
    }

    const messages = document.getElementById("messages");

    messages.innerHTML += `
        <div class="message user">
            ${message}
        </div>
    `;

    input.value = "";

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            message: message
        })
    });

    const data = await response.json();

    messages.innerHTML += `
        <div class="message">
            AI：${data.reply}
        </div>
    `;
}

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return html