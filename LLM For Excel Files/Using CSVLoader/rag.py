from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
import os
from langchain_core.prompts import ChatPromptTemplate
# Define the directory where your vector store is located
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chromadb')

# Load the vector store from the persistent directory
print(f"Loading vector store from {persistent_directory}...")

vector_store = Chroma(persist_directory=persistent_directory)

# Initialize the LLM (Ollama model)
ollama_llm = ChatOllama(model='llama3.2', temperature=0.6, num_gpu=1)

retriever = vector_store.as_retriever()

template = ChatPromptTemplate.from_messages([("system","Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}."),("human","{input}")])

question_answer_chain = create_stuff_documents_chain(ollama_llm, template)
chain = create_retrieval_chain(retriever, question_answer_chain)

query = "What is the gender of Dulce Abril?"
response =chain.invoke({"input": query})

print(response.content)
# # Create the retrieval-based QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=ollama_llm,
#     chain_type="map_reduce",  # You can experiment with different chain types
#     retriever=vector_store.as_retriever()
# )

# # Now, you can query the model with a prompt, and it will use the RAG technique
# query = "What are the details of the person named John Doe?"
# response = qa_chain.run(query)
# print("Response from RAG:", response)