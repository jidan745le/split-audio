#生成字幕时间轴
import speech_recognition as sr
from pydub import AudioSegment
import os

def generate_subtitles(mp3_path, output_srt="output.srt"):
    # 转换MP3为WAV格式
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")

    # 初始化语音识别器
    r = sr.Recognizer()
    
    # 分割音频为30秒片段（可根据需要调整）
    chunk_length_ms = 30000
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    # 生成字幕内容
    subtitles = []
    with open(output_srt, "w", encoding="utf-8") as srt_file:
        for i, chunk in enumerate(chunks):
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            with sr.AudioFile(chunk_path) as source:
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data, language='zh-CN')  # 中文识别
                except sr.UnknownValueError:
                    text = "[无法识别的语音]"
                
                # 计算时间戳
                start_time = i * chunk_length_ms
                end_time = start_time + chunk_length_ms
                
                # 写入SRT格式
                srt_file.write(f"{i+1}\n")
                srt_file.write(f"{ms_to_timestamp(start_time)} --> {ms_to_timestamp(end_time)}\n")
                srt_file.write(f"{text}\n\n")
            
            os.remove(chunk_path)  # 清理临时文件
    
    os.remove(wav_path)  # 清理临时文件
    return output_srt

def ms_to_timestamp(ms):
    seconds = ms // 1000
    milliseconds = ms % 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# 使用示例
generate_subtitles("podcast.mp3", "output.srt")