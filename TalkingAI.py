import os
import time
import streamlit as st
import shutil
import asyncio
import subprocess
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate, # 系统
    HumanMessagePromptTemplate # 用户
)
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
    Microphone
)

load_dotenv()

class ModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"), # 默认会从env中获取名为GROQ_API_KEY的配置作为api key，所以其实也可以不加这行
            temperature=0.1
        )
        system = SystemMessagePromptTemplate.from_template(
                """
                    Your name is Emma.
                    That is very important.
                    Your response must be under 20 words.
                """
            )
        human = HumanMessagePromptTemplate.from_template("{text}")
        self.prompt = ChatPromptTemplate.from_messages([
            system,
            human
        ])
        self.conversation = self.prompt | self.llm

    def process(self, text):
        response = self.conversation.invoke({"text": text})
        return response

class Merge_Transcript:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []
    
    def add_new_sentence(self, sentence):
        self.transcript_parts.append(sentence)

    def get_full_sentence(self):
        return " ".join(self.transcript_parts)

class TextToSpeech:
    Model = "aura-stella-en"

    @staticmethod # 声明静态方法, 检查lib是否安装
    def is_installed(lib_name: str):
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found. if you need to use stream audio, please install it.")
        
        DEEPGRAM_URL = f"{os.environ.get('DEEPGRAM_URL')}/speak?model={self.Model}"
        headers = {
            "Authorization": f"Token {os.environ.get('DEEPGRAM_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {"text": text}

        # 创建子进程
        player_command = [
            "ffplay", 
            "-autoexit", ## 播放结束自动退出
            "-", # 标准输入读取数据
            "-nodisp" # 不要显示图形
        ]
        player_process = subprocess.Popen( #创建一个子进程
            player_command,
            stdin=subprocess.PIPE, # 表示创建的子进程的标准输入从该进程的程序中获取
            stdout=subprocess.DEVNULL, # 不需要，抑制
            stderr=subprocess.DEVNULL
        )

        with requests.post(url=DEEPGRAM_URL, headers=headers, json=payload, stream=True) as request:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    player_process.stdin.write(chunk) # 将deepgram ai中获取到的音频数据写入子进程ffplay中播放
                    player_process.stdin.flush() # 将缓冲区中(flush)的音频数据放到音频标准输入流中，进行下一轮播放
                
        if player_process.stdin: # 判断标准输入流是否存在
            player_process.stdin.close()
        player_process.wait() # 保证主程序等待，直到播放完成



tts = TextToSpeech()


merge_transcript = Merge_Transcript()

async def get_transcript(callback):
    transcription_complete = asyncio.Event() # 使用asyncio的事件，减轻压力。当转译完成，会产生一个事件

    try:
        dg_config = DeepgramClientOptions(
            options={"keepalive": "true"}
        )
        deepgram = DeepgramClient(
            os.getenv("DEEPGRAM_API_KEY"),
            dg_config
        )
        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def message_on(self, result,  **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not result.speech_final: # 谈话是否完成
                merge_transcript.add_new_sentence(sentence)
            else:
                merge_transcript.add_new_sentence(sentence) # 最后一个句子
                full_sentence = merge_transcript.get_full_sentence()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")

                    callback(full_sentence)
                    merge_transcript.reset()

                    transcription_complete.set() # 表示转译完成

        async def error_on(self, error, **kwargs):
            print(f"\n\n{error}\n\n")

        dg_connection.on(LiveTranscriptionEvents.Transcript, message_on)
        dg_connection.on(LiveTranscriptionEvents.Error, error_on)
        options = LiveOptions(
            model="nova-2",
            # language="zh-TW",
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            smart_format=True,
            endpointing=380 #单位：毫秒。当停顿多少时间，表示一段结束。（当语音发送时，并不是一起发的，而是一段一段发）
        )
        await dg_connection.start(options)
        microphone = Microphone(dg_connection.send)
        microphone.start()

        # 此处做无限循环对处理器有压力！
        # while True: 
        #     if not microphone.is_active():
        #         break
        #     await asyncio.sleep(5)
        await transcription_complete.wait()

        microphone.finish()
        await dg_connection.finish()
        
        print("Finished.")

    except Exception as error:
        print(f"Could not open web socket: {error}")
        return

class AiManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = ModelProcessor()   
    

    # 异步处理：语音转文字, llm处理，然后文字转语音，整个过程比较耗时。
    async def start(self):
        st.title("聊天💬机器人🤖")
        st.subheader("你可以向我提问,我会尽量回答你!")

        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        def stream_data(sentence, delay: float = 0.05):
            for char in sentence:
                yield char
                time.sleep(delay)

        # 创建两个占位空间，保证提问和回答只能在这两个区域中，不会出现历史记录的情况
        query_box = st.empty()
        answer_box = st.empty()

        while True:
            await get_transcript(handle_full_sentence)
            if "goodbye" in self.transcription_response.lower():
                with query_box.container(): # 若没有这个的话，就会展示历史记录
                    st.write_stream(
                        stream_data(self.transcription_response)
                    )
                llm_response = self.llm.process(self.transcription_response)
                # print(llm_response.content)
                with answer_box.container():
                    st.write_stream(
                        stream_data(llm_response.content)
                    )
                tts.speak(llm_response.content)

                break
            
            with query_box.container(): # 若没有这个的话，就会展示历史记录
                st.write_stream(
                    stream_data(self.transcription_response)
                )
            llm_response = self.llm.process(self.transcription_response)
            # print(llm_response.content)
            with answer_box.container():
                st.write_stream(
                    stream_data(llm_response.content)
                )

            tts.speak(llm_response.content)

            self.transcription_response = ""

if __name__ == "__main__":
    manager = AiManager()
    asyncio.run(manager.start())