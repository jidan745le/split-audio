import torch
import torchaudio
from pyannote.audio import Pipeline
from pathlib import Path
import datetime
import json

class SpeakerDiarization:
    def __init__(self, auth_token: str, use_gpu: bool = True):
        """初始化说话人分离系统
        
        Args:
            auth_token (str): HuggingFace访问令牌
            use_gpu (bool): 是否使用GPU加速
        """
        # 初始化pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        # 如果可用且需要，使用GPU
        if use_gpu and torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
            print("使用GPU进行处理")
        else:
            print("使用CPU进行处理")

    def process_audio(self, audio_path: str, 
                     num_speakers: int = None,
                     min_speakers: int = None, 
                     max_speakers: int = None) -> dict:
        """处理音频文件
        
        Args:
            audio_path (str): 音频文件路径
            num_speakers (int, optional): 确切的说话人数量
            min_speakers (int, optional): 最少说话人数
            max_speakers (int, optional): 最多说话人数
        
        Returns:
            dict: 处理结果统计
        """
        print(f"\n开始处理音频: {Path(audio_path).name}")
        
        # 从内存加载音频（可能更快）
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 设置处理参数
        params = {"waveform": waveform, "sample_rate": sample_rate}
        
        # 添加说话人数量约束
        if num_speakers:
            params["num_speakers"] = num_speakers
        if min_speakers:
            params["min_speakers"] = min_speakers
        if max_speakers:
            params["max_speakers"] = max_speakers
        
        # 使用进度钩子处理音频
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            diarization = self.pipeline(params, hook=hook)
        
        # 处理结果
        results = self._analyze_diarization(diarization)
        
        # 保存RTTM文件
        rttm_path = Path(audio_path).with_suffix('.rttm')
        with open(rttm_path, "w") as rttm:
            diarization.write_rttm(rttm)
        
        return results

    def _analyze_diarization(self, diarization) -> dict:
        """分析分离结果
        
        Args:
            diarization: PyAnnote分离结果
            
        Returns:
            dict: 统计信息
        """
        stats = {
            'speakers': {},
            'total_duration': 0,
            'total_segments': 0
        }

        tracks = list(diarization.itertracks(yield_label=True))
        
        # 将 tracks 转换为可序列化的格式
        serializable_tracks = []
        for turn, track, speaker in tracks:
            track_dict = {
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            }
            serializable_tracks.append(track_dict)
        
        # 保存可序列化的数据
        with open("diarization.json", "w") as f:
            json.dump(serializable_tracks, f, ensure_ascii=False, indent=2)
        
        # 遍历每个说话片段
        for turn, track, speaker in tracks:
            print(turn,type(turn),list(turn),track,type(track),speaker,type(speaker),turn.start,turn.end)
            duration = turn.end - turn.start
            
            # 更新说话人统计
            if speaker not in stats['speakers']:
                stats['speakers'][speaker] = {
                    'total_time': 0,
                    'segments': 0,
                    'first_appearance': turn.start,
                    'last_appearance': turn.end
                }
            
            speaker_stats = stats['speakers'][speaker]
            speaker_stats['total_time'] += duration
            speaker_stats['segments'] += 1
            speaker_stats['last_appearance'] = turn.end
            
            stats['total_duration'] += duration
            stats['total_segments'] += 1
        
        return stats

    def print_results(self, stats: dict):
        """打印分析结果
        
        Args:
            stats (dict): 统计信息
        """
        print("\n分析结果:")
        print("-" * 50)
        print(f"检测到说话人数量: {len(stats['speakers'])}")
        print(f"总对话时长: {stats['total_duration']:.2f}秒")
        print(f"总片段数: {stats['total_segments']}")
        
        print("\n各说话人统计:")
        for speaker, data in stats['speakers'].items():
            print(f"\n{speaker}:")
            print(f"  说话时长: {data['total_time']:.2f}秒")
            print(f"  片段数量: {data['segments']}")
            print(f"  平均片段时长: {data['total_time']/data['segments']:.2f}秒")
            print(f"  首次出现: {datetime.timedelta(seconds=int(data['first_appearance']))}")
            print(f"  最后出现: {datetime.timedelta(seconds=int(data['last_appearance']))}")

def main():
    # 配置
    AUTH_TOKEN = ""  # 替换为你的HuggingFace令牌
    AUDIO_PATH = "audio.mp3"  # 替换为你的音频文件路径
    
    try:
        # 初始化处理器
        diarization = SpeakerDiarization(AUTH_TOKEN)
        
        # 处理音频（示例：限制说话人数在2-5人之间）
        results = diarization.process_audio(
            AUDIO_PATH,
            min_speakers=2,
            max_speakers=5
        )
        
        # 打印结果
        diarization.print_results(results)
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
