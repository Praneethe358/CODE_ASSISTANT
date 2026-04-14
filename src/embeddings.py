from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("db")

    print("Vector DB created and saved!")
    return vectorstore