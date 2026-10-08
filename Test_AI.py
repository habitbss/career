from openai import OpenAI

# 初始化 DeepSeek 客户端（DeepSeek API 完全兼容 OpenAI 格式）
client = OpenAI(
    api_key="你的DeepSeek_API_Key", base_url="https://api.deepseek.com"
)

# 调用模型回答问题
response = client.chat.completions.create(
    model="deepseek-chat",  # 使用对话模型（若需深度思考可换成 deepseek-reasoner）
    messages=[
        {"role": "system", "content": "你是一个专业的量化交易助手。"},
        {"role": "user", "content": "请用一句话简述什么是主力洗盘。"},
    ],
    stream=False,
)

# 输出结果
print(response.choices[0].message.content)