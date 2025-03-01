from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI

chatglm = ChatOpenAI(model_name="chatglm3-6b",temperature=0)



RetrievalQA.from_chain_type(llm=chatglm,chain_type="stuff",retriever=vector_db)
print(RetrievalQA)