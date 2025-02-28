from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

import torch 
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir,'updated_sample_file.csv')

persistent_directory  = os.path.join(current_dir,'db','chromadb')

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store....")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check your path again."
        )

    csv_loader = CSVLoader(file_path=file_path, csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['first_name', 'last_name', 'Weight','Gender','Country','Age','Date','Id']
})
    docs = csv_loader.load()

    text_splitter = CharacterTextSplitter(separator='\n',chunk_size=1000, chunk_overlap = 0)
    
    split_docs = text_splitter.split_documents(docs)

    print("------Creating Embeddings-------")
    embeddings = OllamaEmbeddings(model = 'llama3.2',num_gpu=1)
    print("\n-------Finished Creating Embeddings------")


    print("\n------Creating vector store------")
    vector_store = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory=persistent_directory  # This ensures that the store is saved to disk
)
    print("\n------Finished Creating Vector Store------")
else:
    print("Vector store already exists. No need to initialise again")

print(f"{current_dir}")
# ollama_llm = ChatOllama(model = 'llama3.2', temperature=0.6, num_gpu=1)

