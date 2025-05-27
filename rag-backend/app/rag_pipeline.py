import os
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import tempfile

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "docs")

llm = Ollama(model="llama3", base_url=OLLAMA_URL)
embedding = OllamaEmbeddings(model="llama3", base_url=OLLAMA_URL)

def ingest_file(filename: str, content: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(content)
        tmp.flush()
        loader = TextLoader(tmp.name)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = Qdrant.from_documents(
        chunks,
        embedding,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    return f"Ingested {len(chunks)} chunks from {filename}"

def query_rag(query: str):
    db = Qdrant(
        collection_name=COLLECTION,
        embedding_function=embedding,
        url=QDRANT_URL
    )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
