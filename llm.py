from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import time
import os
from dotenv import load_dotenv

load_dotenv()


# Ollama
# chat = OllamaLLM(
#     model="openchat:latest",
#     temperature=0
# )

# OpenAI
# chat = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.1,
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url=os.environ.get("CHATGPT_API_ENDPOINT")
# )

# GroqCloud
# 优点：速度快
chat = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1
)

def stream(llm):
    chat = llm

    start_time = time.time()

    system = "你是一个友好的人工助理，最重要的是你总会用中文思考和恢复用户问题"
    human = "写一首赞美{topic}的中文歌曲，不少于1000字"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human)
        ]
    )

    chian = prompt | chat # LCEL
    query = chian.invoke({
        "topic": "月饼"
    })

    print(query.content)

    end_time = time.time()
    print(f"该模型化了{end_time - start_time}秒")


stream(chat)