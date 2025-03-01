import json
import whisper
from pathlib import Path
import torch
import torchaudio
from pyannote.audio import Pipeline
import argparse
from langchain_community.document_loaders import SRTLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os
import sys
import tqdm
from typing import Any, Optional, Mapping

class _CustomProgressBar(tqdm.tqdm):
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

# 设置环境变量
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""  # 替换为你的 token

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
        class CustomProgressHook(ProgressHook):
            def __call__(self, step_name: str, step_artifact: Any,
                    file: Optional[Mapping] = None,
                    total: Optional[int] = None,
                    completed: Optional[int] = None):
                if completed is not None and total is not None:
                    progress = (completed / total) * 100
                    print(f"\r{step_name} 进度: {completed}/{total} ({progress:.1f}%)", end="", flush=True)
                else:
                    # 当没有进度信息时，只显示步骤名称
                    print(f"\r当前步骤: {step_name}", end="", flush=True)
                super().__call__(step_name, step_artifact, file, total, completed)
    
        with CustomProgressHook() as hook:
            diarization = self.pipeline(params, hook=hook)  
        
        # 处理结果
        results = self._analyze_diarization(diarization)    
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
        
        return serializable_tracks


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
    # print(f"正在转录文件: {audio_path}")
    # def progress_callback(step, total):
    #     print(f"\r转录音频进度: {step}/{total} ({(step/total*100):.1f}%)", end="")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    return result


def merge_same_speaker_segments(diarization_segments, merge_threshold=1):
    """
    将 PyAnnote 连续且说话人相同的相邻分段进行合并。
    
    Args:
        diarization_segments (list): PyAnnote 输出的说话人分段列表
        merge_threshold (float): 若相邻分段之间的时间差小于此阈值且说话人相同，则可合并。
    
    Returns:
        list: 合并后的分段列表
    """
    # 根据开始时间排序
    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
    
    merged = []
    for seg in diarization_segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            # 如果说话人相同 且 这两个分段的起止时间相差不大，则合并
            if (last["speaker"] == seg["speaker"] 
                and abs(seg["start"] - last["end"]) <= merge_threshold):
                # 更新 merged 中最后一段的 end
                merged[-1]["end"] = max(last["end"], seg["end"])
            else:
                merged.append(seg)
    return merged


def combine_whisper_diarization_with_ratio(whisper_data, diarization_data, overlap_threshold=0.3):
    """
    将 Whisper 转录与 PyAnnote Diarization 数据合并，按说话人输出段落级结果。
    仅当单词与分段的相交时长/单词时长 >= overlap_threshold 时才纳入。
    
    Args:
        whisper_data (dict): Whisper 输出 (包含 text, segments, 及 words)
        diarization_data (list): PyAnnote 说话人分段
        overlap_threshold (float): 重叠阈值（0~1之间），如 0.3 表示重叠达到单词时长的 30% 以上则纳入。
    
    Returns:
        list: 类似 [
            {
              "speaker": "SPEAKER_00",
              "start": 17.16,
              "end": 25.00,
              "text": "此时间段内说话人全部文字",
              "words": [
                {"start": 17.16, "end": 17.28, "word": "xxx"},
                ...
              ]
            },
            ...
        ]
    """
    # 1) 先合并 PyAnnote 中相邻、同 speaker 的分段 (可选)
    merged_segments = merge_same_speaker_segments(diarization_data)
    
    # 2) 收集所有 words
    all_words = []
    for seg in whisper_data["segments"]:
        for w in seg.get("words", []):
            all_words.append(w)
    
    combined_result = []
    
    # 3) 遍历每个说话人分段
    for seg in merged_segments:
        spk = seg["speaker"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        segment_words = []
        
        # 在此段时间内，筛选出满足"交集时长/单词时长 >= overlap_threshold"的 words
        for w in all_words:
            w_start = w["start"]
            w_end = w["end"]
            word_duration = w_end - w_start
            if word_duration <= 0:
                # 规避异常情况
                continue
            
            # 计算交集
            intersect_start = max(w_start, seg_start)
            intersect_end = min(w_end, seg_end)
            intersect_len = intersect_end - intersect_start
            
            # 若无交集
            if intersect_len <= 0:
                continue
            
            overlap_ratio = intersect_len / word_duration
            # 只有当这个比例 >= overlap_threshold，才视作该单词属于该说话人分段
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


# 添加参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description='音频处理程序')
    parser.add_argument('--audio_path', type=str, required=True, help='音频文件路径')
    return parser.parse_args()

def save_as_srt(combined_results, output_path):
    """将合并结果保存为 SRT 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(combined_results, 1):
            # 转换时间戳为 SRT 格式 (HH:MM:SS,mmm)
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            
            # 写入 SRT 格式
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['speaker']}: {segment['text']}\n\n")

def format_timestamp(seconds):
    """将秒数转换为 SRT 时间戳格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def build_vector_store(srt_path):
    """构建向量存储"""
    # 加载 SRT 文件
    import pysrt

    parsed_info = pysrt.open(srt_path)
    docs = []
    for index, subtitle in enumerate(parsed_info, start=1):
        # 获取起止时间
        start = subtitle.start.to_time()
        end = subtitle.end.to_time()

        # SRTLoader 里不一定非要存时间戳，你可以自定义
        metadata = {
            "start": str(start),
            "end": str(end),
            "sequence": subtitle.index or index,
            "source": srt_path,
        }
        # 创建 Document
        docs.append(
                Document(
                    page_content=subtitle.text.strip(),
                    metadata=metadata
                )
            )   
    
    # 2) 使用 TextSplitter 对 documents 进行分块
    #    - chunk_size: 每个文本块的最大字符数
    #    - chunk_overlap: 不同 chunk 间的重叠字符数（可视需要调整）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)
    
    # 将分块后的 Document 对象转换为可序列化的字典（如果需要查看或存储）
    serializable_docs = []
    for doc in split_docs:
        doc_dict = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        serializable_docs.append(doc_dict)
    
    # 保存可序列化的文档
    with open("document.json", "w") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    
    # 创建嵌入模型
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True, 'device': 'cuda'},  # GPU可用时改为'cuda'
        encode_kwargs={'normalize_embeddings': True}
    )

    
    # 创建向量存储
    vectorstore = FAISS.from_documents(
        documents=split_docs,  # 使用原始 documents
        embedding=embeddings
    )
    
    return vectorstore

def search_content(vectorstore, query, k=5):
    """搜索相关内容"""
    results = vectorstore.similarity_search(query, k=k)
    
    # 将 Document 对象转换为可序列化的字典
    serializable_results = []
    for doc in results:
        doc_dict = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        serializable_results.append(doc_dict)
    
    return serializable_results

def build_qa_chain(vectorstore):
    """构建问答链"""
    # 创建提示模板
    prompt_template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文: {context}

问题: {question}

答案:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 创建 LLM
    llm = HuggingFaceEndpoint(
        repo_id="THUDM/chatglm3-6b",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        task="text-generation",
        temperature=0.5,  # 直接指定 temperature
        model_kwargs={
            "max_length": 512
        }
    )

    # 创建 QA 链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def ask_question(qa_chain, question):
    """提问并获取答案"""
    # 使用 invoke 替代直接调用
    result = qa_chain.invoke({"query": question})
    
    # 构建可序列化的响应
    response = {
        "answer": result["result"],
        "source_documents": []
    }
    
    # 添加源文档信息，处理可能的键名差异
    for doc in result["source_documents"]:
        source_doc = {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        response["source_documents"].append(source_doc)
    
    return response

if __name__ == "__main__":
    args = parse_arguments()
    audio_path = args.audio_path
    
    # 获取音频路径
    model_name = "small"   # 使用固定值
    
    # 执行转录
    # whisper_data = transcribe_audio(audio_path, model_name)


    # 配置
    AUTH_TOKEN = ""  # 替换为你的HuggingFace令牌
    AUDIO_PATH = audio_path  # 替换为你的音频文件路径
    
    # 初始化处理器
    diarization = SpeakerDiarization(AUTH_TOKEN)
        
    # 处理音频（示例：限制说话人数在2-5人之间）
    diarization_data = diarization.process_audio(
            AUDIO_PATH,
            min_speakers=2,
            max_speakers=2
        )
    
    # 以30%为阈值
    overlap_threshold = 0.3  
    combined_results = combine_whisper_diarization_with_ratio(
        whisper_data, 
        diarization_data,
        overlap_threshold=overlap_threshold
    )
    
    print("\n--- 最终基于说话人的转录 (交集占比 >= 30% 才纳入) ---")
    print(json.dumps(combined_results, indent=2, ensure_ascii=False))

    # # 保存 JSON 结果
    json_path = audio_path.replace(".mp3", "_merge.json")
    # with open(json_path, "w", encoding='utf-8') as f:
    #     json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    # # 保存 SRT 文件
    srt_path = audio_path.replace(".mp3", ".srt")
    save_as_srt(combined_results, srt_path)
    
    # 构建向量存储
    vectorstore = build_vector_store(srt_path)
    # vectorstore = build_vector_store("breakup.srt")
    # print(vectorstore,"ddd")
        # 1. 查看存储的文档数量
    print("\n1. 存储的文档数量:")
    print(f"Total documents: {len(vectorstore.docstore._dict)}")
    
    # 2. 查看文档内容示例
    print("\n2. 文档内容示例:")
    for doc_id, doc in list(vectorstore.docstore._dict.items())[:3]:  # 只显示前3个
        print(f"\nDocument ID: {doc_id}")
        print(f"Content: {doc.page_content[:100]}...")  # 显示前100个字符
        print(f"Metadata: {doc.metadata}")
    
    # 3. 查看向量维度
    print("\n3. 向量维度:")
    if hasattr(vectorstore, 'index'):
        print(f"Vector dimension: {vectorstore.index.d}")
    
    # 1. 获取 FAISS 索引
    index = vectorstore.index
    
    # 2. 获取所有向量
    if hasattr(index, 'reconstruct'):
        print("\n向量数据:")
        total_vectors = index.ntotal  # 向量总数
        print(f"向量总数: {total_vectors}")
        
        # 显示前几个向量的示例
        for i in range(min(3, total_vectors)):
            vector = index.reconstruct(i)  # 获取第i个向量
            print(f"\n向量 {i} 的前10个维度:")
            print(vector[:10])  # 只显示前10个维度
            
            # 获取对应的文档
            doc_id = list(vectorstore.docstore._dict.keys())[i]
            doc = vectorstore.docstore._dict[doc_id]
            print(f"对应文档内容: {doc.page_content[:50]}...")
    
    # 构建 QA 链
    qa_chain = build_qa_chain(vectorstore)
    
    # 示例问题
    question = "philosophy meaning of breakup？"
    response = ask_question(qa_chain, question)
    
    # 保存问答结果
    with open("qa_result.json", "w", encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    
    # 打印答案
    print("\n问题:", question)
    print("\n答案:", response["answer"])
    print("\n相关片段:")
    for doc in response["source_documents"]:
        # 检查并使用正确的时间戳键名
        start_time = doc['metadata'].get('start', doc['metadata'].get('start_time', 'N/A'))
        end_time = doc['metadata'].get('end', doc['metadata'].get('end_time', 'N/A'))
        print(f"时间: {start_time} - {end_time}")
        print(f"内容: {doc['content']}\n")

    # 保存向量存储（可选）
    vectorstore.save_local(audio_path.replace(".mp3", "_vectors"))
    
    print(f"处理完成：\nJSON: {json_path}\nSRT: {srt_path}")

    related_docs = search_content(vectorstore, "philosophy meaning of breakup")

    # 现在可以保存为 JSON
    with open("related_docs.json", "w") as f:
        json.dump(related_docs, f, ensure_ascii=False, indent=2)
