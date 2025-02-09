import os
import asyncio
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient, # 客户端
    DeepgramClientOptions, 
    LiveTranscriptionEvents, # 转译事件，用于处理转译过程的回传及结果，错误等
    LiveOptions,
    Microphone
)

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")

class Merge_Transcript:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_new_sentence(self, sentence):
        self.transcript_parts.append(sentence)

    def get_full_sentence(self):
        return ','.join(self.transcript_parts)

merge_transcript = Merge_Transcript()

# 麦克风语音转文字功能
async def get_transcript():
    try:
        df_config = DeepgramClientOptions(
            options={
                "keepalive": "true" # 避免聊天过程中停顿导致连接中断
            }
        )
        deepgram = DeepgramClient(API_KEY, df_config)
        dg_connection = deepgram.listen.asynclive.v("1")

        async def message_on(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            # print(sentence)

            if not result.speech_final:
                merge_transcript.add_new_sentence(sentence)
            else:
                merge_transcript.add_new_sentence(sentence)
                full_sentence = merge_transcript.get_full_sentence()
                print(f"Speaker: {full_sentence}")

                merge_transcript.reset()

        async def error_on(self, error, **kwargs):
            print(f"\n\n{error}\n\n")

        dg_connection.on(LiveTranscriptionEvents.Transcript, message_on)
        dg_connection.on(LiveTranscriptionEvents.Error, error_on)

        options = LiveOptions(
            model="nova-2",
            language="zh-TW",
            # language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            smart_format=True,
            endpointing=380 #单位：毫秒。当停顿多少时间，表示一段结束。（当语音发送时，并不是一起发的，而是一段一段发）
        )
        await dg_connection.start(options)

        microphone = Microphone(dg_connection.send)
        microphone.start()

        while True:
            if not microphone.is_active():
                break
            await asyncio.sleep(1) # 每隔一秒检测麦克风检测, ”await“用于等待其他异步任务完成操作。
        microphone.finish()

        dg_connection.finish()
        print("Finished")


    except Exception as error:
        print(f"Failed to connected: {error}")
        return


asyncio.run(get_transcript())

