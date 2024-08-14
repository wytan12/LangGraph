from langchain_community.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

from LangGraph.config import azure_embeddings


def process_document(file_path, file_type):
    if file_type == "docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    docs = text_splitter.split_documents(documents)
    return docs


docs_folder = "docs"

all_docs = []

for filename in os.listdir(docs_folder):
    file_path = os.path.join(docs_folder, filename)
    file_type = filename.split('.')[-1]

    chunks = process_document(file_path, file_type)
    all_docs.extend(chunks)

db = FAISS.from_documents(all_docs, azure_embeddings)

retriever = db.as_retriever()