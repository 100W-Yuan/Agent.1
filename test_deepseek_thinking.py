import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY"
    )


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": "请回答：25乘以4等于多少？"
        }
    ],
    extra_body={
        "thinking": {
            "type": "enabled"
        }
    }
)


message = response.choices[0].message


print("========== DeepSeek Thinking 测试 ==========")

print(
    "content:"
)

print(
    message.content
)

print()

print(
    "reasoning_content 是否存在：",
    bool(
        getattr(
            message,
            "reasoning_content",
            None
        )
    )
)

print()

print(
    "reasoning_content："
)

print(
    getattr(
        message,
        "reasoning_content",
        None
    )
)