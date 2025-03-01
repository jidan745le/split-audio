import whisper
import sys
from pathlib import Path
import json
import torch

def transcribe_audio(audio_path: str, model_name: str = "base") -> dict:
    """
    使用 Whisper 模型转录音频文件
    
    Args:
        audio_path: 音频文件路径
        model_name: Whisper 模型名称 (tiny, base, small, medium, large)
    
    Returns:
        包含转录结果的字典
    """
    # 检查 CUDA 环境
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前 CUDA 设备: {torch.cuda.get_device_name()}")
    
    # 检查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"正在加载 Whisper {model_name} 模型...")
    model = whisper.load_model(model_name).to(device)
    
    # 转录音频
    print(f"正在转录文件: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    return result

def save_transcription(result: dict, output_path: str):
    """
    保存转录结果到文本文件
    
    Args:
        result: Whisper 转录结果
        output_path: 输出文件路径
    """
    txt_path = Path(output_path).with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 写入完整文本
        f.write("完整文本：\n")
        f.write(result['text'] + "\n\n")
        
        # 写入单词级详细信息
        f.write("单词级转录：\n")
        for i, segment in enumerate(result['segments'], 1):
            f.write(f"\n段落 {i}（{segment['start']:.2f}-{segment['end']:.2f}秒）:\n")
            for word in segment['words']:
                f.write(f"{word['start']:.2f}-{word['end']:.2f}: {word['word'].strip()}\n")
    
    print(f"已保存详细转录结果到: {txt_path}")
    
    # 保存带时间戳的字幕文件
    srt_path = Path(output_path).with_suffix('.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            # 转换时间戳为 SRT 格式
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            
            # 写入 SRT 格式
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    print(f"已保存字幕文件到: {srt_path}")

def format_timestamp(seconds: float) -> str:
    """
    将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    # if len(sys.argv) < 2:
    #     print("使用方法: python whisper.py <音频文件路径> [模型名称]")
    #     sys.exit(1)
    
    audio_path = "audio.mp3"
    model_name = "small"
    
    if not Path(audio_path).exists():
        print(f"错误: 文件 {audio_path} 不存在")
        sys.exit(1)
    
    # 执行转录
    result = transcribe_audio(audio_path, model_name)
    
    # 保存完整的 JSON 结果
    output_base = Path(audio_path).stem
    json_path = Path(output_base).with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已保存完整转录结果到: {json_path}")
    
    # 保存结果
    save_transcription(result, output_base)

if __name__ == "__main__":
    main()
