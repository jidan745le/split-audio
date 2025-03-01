from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SRTLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json

def build_vector_store(srt_path):
    """构建向量存储（带文本切割）"""
    # 1) 加载 SRT 文件，得到包含所有字幕段的 documents
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
    
    # 若想持久化所有分块后的文本，可以在此将 serializable_docs 写出为 JSON
    with open("document1.json", "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    
    # 3) 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True, 'device': 'cuda'},  
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 4) 使用分块后的文档来创建向量存储
    vectorstore = FAISS.from_documents(
        documents=split_docs,  
        embedding=embeddings
    )
    
    return vectorstore

build_vector_store("test.srt")