import json
import whisper
import torch
import torchaudio
from pyannote.audio import Pipeline
import os
from typing import Dict, Any, Optional, Mapping
from dotenv import load_dotenv
from tqdm import tqdm
import sys

# 加载 .env 文件
load_dotenv()

class SpeakerDiarization:
    def __init__(self, socketio=None):
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
        self.socketio = socketio

    def process_audio(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 4) -> dict:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        params = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers
        }
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        class CustomProgressHook(ProgressHook):
            def __init__(self, socketio=None):
                super().__init__()
                self.socketio = socketio
                
            def __call__(self, step_name: str, step_artifact: Any,
                    file: Optional[Mapping] = None,
                    total: Optional[int] = None,
                    completed: Optional[int] = None):
                if completed is not None and total is not None:
                    progress = (completed / total) * 100
                    progress_msg = f"{step_name} 进度: {completed}/{total} ({progress:.1f}%)"
                    print(f"\r{progress_msg}", flush=True)
                    
                    # 通过WebSocket发送进度信息
                    if self.socketio:
                        self.socketio.emit('diarization_progress', {
                            'step': step_name,
                            'completed': completed,
                            'total': total,
                            'progress': round(progress, 1),
                            'message': progress_msg
                        })
                else:
                    # 当没有进度信息时，只显示步骤名称
                    print(f"\r当前步骤: {step_name}", flush=True)
                    if self.socketio:
                        self.socketio.emit('diarization_progress', {
                            'step': step_name,
                            'message': f"当前步骤: {step_name}"
                        })
                super().__call__(step_name, step_artifact, file, total, completed)
    
        with CustomProgressHook(self.socketio) as hook:
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

def process_audio_file(audio_path: str, socketio=None) -> Dict[str, Any]:
    # 自定义进度条，通过WebSocket发送进度
    class _CustomProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._current = self.n  # 设置初始值
            self.socketio = socketio

        def update(self, n):
            super().update(n)
            self._current += n
            
            # 计算进度百分比
            if self.total is not None:
                progress = (self._current / self.total) * 100
                progress_msg = f"Whisper进度: {self._current}/{self.total} ({progress:.1f}%)"
                print(f"\r{progress_msg}", flush=True)
                
                # 通过WebSocket发送进度信息
                if self.socketio:
                    self.socketio.emit('whisper_progress', {
                        'completed': self._current,
                        'total': self.total,
                        'progress': round(progress, 1),
                        'message': progress_msg
                    })
    
    import whisper.transcribe 
    transcribe_module = sys.modules['whisper.transcribe']
    transcribe_module.tqdm.tqdm = _CustomProgressBar
    
    # 发送开始处理的消息
    if socketio:
        socketio.emit('processing_status', {'status': 'started', 'message': '开始处理音频文件'})
    
    # 1. 使用Whisper进行转录（强制使用CPU）
    if socketio:
        socketio.emit('processing_status', {'status': 'whisper_loading', 'message': '加载Whisper模型'})
    
    model = whisper.load_model("base").cpu()
    
    if socketio:
        socketio.emit('processing_status', {'status': 'whisper_transcribing', 'message': '开始转录音频'})
    
    # 设置tqdm的socketio
    transcribe_module.tqdm.tqdm.socketio = socketio
    whisper_data = model.transcribe(audio_path, word_timestamps=True)
    
    # 2. 进行说话人分离
    if socketio:
        socketio.emit('processing_status', {'status': 'diarization_started', 'message': '开始说话人分离'})
    
    diarization = SpeakerDiarization(socketio)
    diarization_data = diarization.process_audio(audio_path)
    
    # 3. 合并结果
    if socketio:
        socketio.emit('processing_status', {'status': 'combining_results', 'message': '合并转录和说话人分离结果'})
    
    combined_results = combine_whisper_diarization_with_ratio(
        whisper_data,
        diarization_data,
        overlap_threshold=0.3
    )
    
    # 发送处理完成的消息
    if socketio:
        socketio.emit('processing_status', {'status': 'completed', 'message': '音频处理完成'})
    
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
        if len(segment_words) > 0:
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