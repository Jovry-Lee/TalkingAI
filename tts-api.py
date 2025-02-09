# Text To Speech (Api Request)


import os
from dotenv import load_dotenv
import requests # 用于向deepgram发送请求

load_dotenv()

dg_api_key = os.environ.get("DEEPGRAM_API_KEY")

model = "aura-stella-en"
url = f"{os.environ.get('DEEPGRAM_URL')}/speak?model={model}"

headers = {
    "Authorization": f"Token {dg_api_key}",
    "Content-Type": "application/json"
}

payload = {
    "text": "Hello, Joey, How are you? My name is Emma and I'm very glad to meet you. What do you think of the Text-To-Speech API?"
}

print(url, headers, payload)

response = requests.post(
    url, 
    headers=headers, 
    json=payload,
    stream=True
)

audio_file_path = "output-stream.wav"
if response.status_code == 200:
    # 将语音转换为音频文件
    with open(audio_file_path, "wb") as f:
        # f.write(response.content)
        for chunk in response.iter_content(chunk_size=1024):
            if chunk: # 最后一个分块不存在
                f.write(chunk)
        print("File save successful!")
else:
    print(f"Error: {response.status_code} - {response.text}")