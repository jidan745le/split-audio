from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter

from langchain_community.document_loaders import SRTLoader
from langchain_community.vectorstores import FAISS,Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_hub
# from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import uuid


# 测试文本（约200字符，包含重复结构）
test_text = '''
深度学习模型在处理长文本时，需要有效的分块策略，人工智能模型。当文本被适当分割时，可以保留上下文信息，人工智能模型。递归字符分割器的核心机制是：优先用大分隔符拆分，若块太大则继续用次级分隔符分割，人工智能模型。这个过程会重复直到所有块都小于目标尺寸，人工智能模型。
'''

# 建议参数设置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "，", "。", " ", ""]  # 注意逗号在句号前
)

# 执行分割
text_chunks = text_splitter.split_text(test_text)
# text_chunks_document = text_splitter.split_documents(test_text)

# 查看结果
for i, chunk in enumerate(text_chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):\n{chunk}\n{'-'*50}")

# for i, chunk in enumerate(text_chunks_document):
#     print(f"Chunk {i+1} ({len(chunk)} chars):\n{chunk}\n{'-'*50}")

# text_splitter_normal = CharacterTextSplitter(
#     separator="。",
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
# )

# text_chunks_normal = text_splitter_normal.split_text(test_text)
# print("************")
# for i, chunk in enumerate(text_chunks_normal):
#     print(f"Chunk {i+1} ({len(chunk)} chars):\n{chunk}\n{'-'*50}")

# 初始化Nomic嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={'trust_remote_code': True, 'device': 'cpu'},  # GPU可用时改为'cuda'
    encode_kwargs={'normalize_embeddings': True}
)

# 将文本块转换为向量
vectors = embedding_model.embed_documents(text_chunks)

# 创建FAISS索引
dimension = 768  # nomic-embed-text-v1的输出维度
index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
index.add(np.array(vectors).astype('float32'))

# 创建文档存储（示例）
documents = [
    Document(
        page_content=chunk,        
        metadata={"source": "podcast", "chunk_id": str(uuid.uuid4())}
       
    ) for chunk in text_chunks
]

print(documents)

# 配置FAISS参数
your_docstore = InMemoryDocstore({
    doc.metadata["chunk_id"]: doc for doc in documents
})

your_mapping = {
    i: doc.metadata["chunk_id"] for i, doc in enumerate(documents)
}

print(your_mapping,your_docstore)

vector_db = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=your_docstore,       # 原始文本存储器
    index_to_docstore_id=your_mapping,  # 索引到文档ID的映射
)


# 查询示例
query = "人工智能模型的分块策略"
query_vector = embedding_model.embed_query(query)
# print(query_vector)
D, I = index.search(np.array([query_vector]).astype('float32'), k=2)
print(D,I)
vector_db.similarity_search_with_score(query,k=2)

# 打印相似结果
print("最相关的3个文本块：")
for idx in I[0]:
    print(f"块 {idx+1}: {text_chunks[idx][:50]}...")

# 查看前5个文档的元数据和内容摘要
for doc_id in list(vector_db.docstore._dict.keys())[:5]:
    doc = vector_db.docstore._dict[doc_id]
    print(f"文档ID: {doc_id}")
    print(f"文档: {doc}")
    print(f"元数据: {doc.metadata}")
    print(f"内容摘要: {doc.page_content[:50]}...\n")


