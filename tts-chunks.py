import re #正则表达式
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

input_text = "Our story begins in a peaceful woodland kingdom where a lively squirrel named Frolic made his abode high up within a cedar tree's embrace. He was not a usual woodland creature, for he was blessed with an insatiable curiosity and a heart for adventure. Nearby, a glistening river snaked through the landscape, home to a wonder named Splash - a silver-scaled flying fish whose ability to break free from his water-haven intrigued the woodland onlookers. This magical world moved on a rhythm of its own until an unforeseen circumstance brought Frolic and Splash together. One radiant morning, while Frolic was on his regular excursion, and Splash was making his aerial tours, an unpredictable wave playfully tossed and misplaced Splash onto the riverbank. Despite his initial astonishment, Frolic hurriedly and kindly assisted his new friend back to his watery abode. Touched by Frolic's compassion, Splash expressed his gratitude by inviting his friend to share his world. As Splash perched on Frolic's back, he tasted of the forest's bounty, felt the sun’s rays filter through the colors of the trees, experienced the conversations amidst the woods, and while at it, taught the woodland how to blur the lines between earth and water."

# 讲一大段文章进行分段(.!?为一句话的结尾)
def segment_text_by_sentence(text):
    sentence_boundaries = re.finditer(r'(?<=[.!?])\s+', text)
    boundaries_indices = [boundary.start() for boundary in sentence_boundaries]
    print(boundaries_indices)

    segments = []
    start = 0

    for boundary_index in boundaries_indices:
        segments.append(text[start:boundary_index].strip())
        start = boundary_index
    segments.append(text[start:].strip())

    return segments

def synthesize_audio(chunk, output_file):
    payload = {"text": chunk}
    with requests.post(url, headers=headers, json=payload, stream=True) as request:
        for chunk in request.iter_content(chunk_size=1024):
            if chunk:
                output_file.write(chunk)

def main():
    segments = segment_text_by_sentence(input_text)
    # print(segments)

    audio_file_path = "output-segments.wav"
    with open(audio_file_path, "wb") as f:
        for segment_text in segments:
            synthesize_audio(segment_text, f)
    print("Audio file creation completed!")

if __name__ == "__main__":
    main()