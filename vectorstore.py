from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

### Documents DB
def load_docs(files):
    for file in files:
        yield from PyPDFLoader(file).load()

def split_docs(docs, chunk_size=1200, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for doc in docs:
        yield from text_splitter.split_documents([doc])

def add_files(files,
              collection_name,
              persist_directory = ".db",
              embeddings_model="dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn"):
    docs = load_docs(files)
    doc_splits = split_docs(docs)

    embedding = SentenceTransformerEmbeddings(model_name=embeddings_name)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return

def get_vectorstore(collection_name, persist_directory = ".db", embeddings_model="dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn"):
    vectorstore = Chroma(collection_name=collection_name, persist_directory=persist_directory,
                         embedding_function=SentenceTransformerEmbeddings(model_name=embeddings_model))
    return vectorstore    