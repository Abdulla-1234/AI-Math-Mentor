import os
from dotenv import load_dotenv

# Load environment variables (Still good to have for later when we use LLMs)
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
KB_DIR = "data/knowledge_base"
VECTOR_DB_PATH = "data/faiss_index"

def build_vector_store():
    """
    Loads documents from the knowledge base, chunks them, embeds them locally, 
    and saves a FAISS vector index.
    """
    print("Loading documents...")
    loader = DirectoryLoader(KB_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found in the knowledge base directory!")
        return None

    print(f"Loaded {len(documents)} documents. Chunking...")
    # Chunk the documents into smaller pieces 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50  
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks. Embedding and saving to FAISS...")
    
    # PLAN B: Use free, local HuggingFace embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save it locally 
    vectorstore.save_local(VECTOR_DB_PATH)
    print("Vector store built and saved successfully!")
    return vectorstore

def get_retriever():
    """
    Loads the saved FAISS index and returns a retriever object.
    """
    # We must use the exact same embedding model to search as we used to build it
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(VECTOR_DB_PATH):
        print("Vector database not found. Building it now...")
        build_vector_store()
        
    try:
        # Load the existing vector store
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        # Retrieve the top 3 most relevant chunks
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def retrieve_context(query: str):
    """
    Retrieves relevant context for a query. Returns formatted sources to avoid hallucinations.
    """
    retriever = get_retriever()
    if not retriever:
        return "System Error: Knowledge base unavailable."

    docs = retriever.invoke(query)
    
    if not docs:
        # Strict rule: No hallucinated citations if retrieval fails
        return "No relevant context found in the knowledge base."
    
    # Format the retrieved documents clearly
    context_str = "Retrieved Context:\n"
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        context_str += f"\n--- Source {i+1}: {source} ---\n{doc.page_content}\n"
        
    return context_str

# --- Quick local test ---
if __name__ == "__main__":
    test_query = "What is the formula for probability?"
    print(f"\nSearching for: '{test_query}'\n")
    context = retrieve_context(test_query)
    print(context)