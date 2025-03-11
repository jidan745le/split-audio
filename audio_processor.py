import json
import whisper
import torch
import torchaudio
from pyannote.audio import Pipeline
import os
from typing import Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
import sys
from typing import Optional, Mapping, Any

# 加载 .env 文件
load_dotenv()

class _CustomProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value
        
    def update(self, n):
        super().update(n)
        self._current += n
        
        # Handle progress here
        print("\nProgress~: " + str(self._current) + "/" + str(self.total))

import whisper.transcribe 
transcribe_module = sys.modules['whisper.transcribe']
transcribe_module.tqdm.tqdm = _CustomProgressBar

class SpeakerDiarization:
    def __init__(self):
        # 从环境变量获取 token
        auth_token = os.getenv('HUGGINGFACE_TOKEN')
        if not auth_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
            
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        # 强制使用CPU
        self.pipeline.to(torch.device("cpu"))

    def process_audio(self, audio_path: str, min_speakers: int = 2, max_speakers: int = 2) -> dict:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        params = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers
        }
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        class CustomProgressHook(ProgressHook):
            def __call__(self, step_name: str, step_artifact: Any,
                    file: Optional[Mapping] = None,
                    total: Optional[int] = None,
                    completed: Optional[int] = None):
                if completed is not None and total is not None:
                    progress = (completed / total) * 100
                    print(f"\r{step_name} 进度: {completed}/{total} ({progress:.1f}%)",flush=True)
                else:
                    # 当没有进度信息时，只显示步骤名称
                    print(f"\r当前步骤: {step_name}",flush=True)
                super().__call__(step_name, step_artifact, file, total, completed)
    
        with CustomProgressHook() as hook:
            diarization = self.pipeline(params, hook=hook)
        return self._analyze_diarization(diarization)

    def _analyze_diarization(self, diarization) -> dict:
        tracks = list(diarization.itertracks(yield_label=True))
        serializable_tracks = []
        for turn, track, speaker in tracks:
            track_dict = {
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            }
            serializable_tracks.append(track_dict)
        return serializable_tracks

def process_audio_file(audio_path: str) -> Dict[str, Any]:
    # 不再需要在这里设置 token
    
    # 1. 使用Whisper进行转录（强制使用CPU）
    model = whisper.load_model("base").cpu()
    whisper_data = model.transcribe(audio_path, word_timestamps=True)
    
    # 2. 进行说话人分离
    diarization = SpeakerDiarization()
    diarization_data = diarization.process_audio(audio_path)
    
    # 3. 合并结果
    combined_results = combine_whisper_diarization_with_ratio(
        whisper_data,
        diarization_data,
        overlap_threshold=0.3
    )
    
    return combined_results

def handle_overlapping_segments(diarization_data):
    """处理重叠的说话人片段"""
    # 按开始时间排序
    sorted_segments = sorted(diarization_data, key=lambda x: x["start"])
    
    # 创建一个新的无重叠片段列表
    non_overlapping = []
    
    for segment in sorted_segments:
        if not non_overlapping:
            # 第一个片段直接添加
            non_overlapping.append(segment)
            continue
        
        last = non_overlapping[-1]
        
        # 检查是否有重叠
        if segment["start"] < last["end"]:
            # 有重叠，根据持续时间决定保留哪个
            current_duration = segment["end"] - segment["start"]
            last_duration = last["end"] - last["start"]
            
            # 如果当前片段更长，替换最后一个片段
            if current_duration > last_duration:
                non_overlapping[-1] = segment
            # 否则保持不变
        else:
            # 无重叠，直接添加
            non_overlapping.append(segment)
    
    return non_overlapping

def combine_whisper_diarization_with_ratio(whisper_data, diarization_data, overlap_threshold=0.3):
    # 将原有的合并函数代码复制到这里
    # 合并相邻的相同说话人片段
    merged_segments = merge_same_speaker_segments(handle_overlapping_segments(diarization_data))
    
    # 收集所有words
    all_words = []
    for seg in whisper_data["segments"]:
        for w in seg.get("words", []):
            all_words.append(w)
    
    combined_result = []
    
    # 遍历每个说话人分段
    for seg in merged_segments:
        spk = seg["speaker"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        segment_words = []
        
        # 筛选满足条件的words
        for w in all_words:
            w_start = w["start"]
            w_end = w["end"]
            word_duration = w_end - w_start
            if word_duration <= 0:
                continue
            
            intersect_start = max(w_start, seg_start)
            intersect_end = min(w_end, seg_end)
            intersect_len = intersect_end - intersect_start
            
            if intersect_len <= 0:
                continue
            
            overlap_ratio = intersect_len / word_duration
            if overlap_ratio >= overlap_threshold:
                segment_words.append({
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "word": w["word"].strip()
                })
        
        # 拼接文本
        segment_text = " ".join(w["word"] for w in segment_words)
        
        combined_result.append({
            "speaker": spk,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": segment_text,
            "words": segment_words
        })
        
    return combined_result

def merge_same_speaker_segments(diarization_segments, merge_threshold=1):
    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
    
    merged = []
    for seg in diarization_segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if (last["speaker"] == seg["speaker"] 
                and abs(seg["start"] - last["end"]) <= merge_threshold):
                merged[-1]["end"] = max(last["end"], seg["end"])
            else:
                merged.append(seg)
    return merged 