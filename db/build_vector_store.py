import os
import sqlite3
import pypdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def get_embeddings(provider="local"):
    """
    Returns the appropriate embeddings model based on the provider.
    Toggles between 'local' HuggingFace and 'google' Gemini.
    """
    if provider == "google":
        print("Using Google Gemini Embeddings (models/text-embedding-004)...")
        # Ensure API key is set
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is not set in your environment variables.")
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        print("Using Local HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_and_chunk_pdf(pdf_path):
    print(f"Reading PDF from {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    documents = []
    
    # Extract text from each page and attach metadata
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            metadata = {
                "source": os.path.basename(pdf_path),
                "page": i + 1  # 1-based page index
            }
            # Create a LangChain document representation
            from langchain_core.documents import Document
            documents.append(Document(page_content=page_text, metadata=metadata))
            
    print(f"Extracted {len(documents)} pages from the PDF.")
    
    # Split text into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split pages into {len(chunks)} text chunks.")
    return chunks

def build_store(provider="local"):
    pdf_path = "handbook.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file {pdf_path} not found in the workspace directory.")
        
    # Get chunks
    chunks = load_and_chunk_pdf(pdf_path)
    
    # Get embeddings
    embeddings = get_embeddings(provider)
    
    # Build FAISS index
    print("Building FAISS index...")
    if provider == "google":
        # Google has free-tier rate limits (100 requests per minute).
        # We will build incrementally in batches of 50, sleeping 60 seconds in between.
        batch_size = 50
        print(f"Building incrementally for Google embeddings (batch size: {batch_size}, delay: 60s)...")
        db = FAISS.from_documents(chunks[:batch_size], embeddings)
        import time
        for i in range(batch_size, len(chunks), batch_size):
            print(f"Sleeping 60 seconds to reset rate limit...")
            time.sleep(60)
            batch = chunks[i:i+batch_size]
            print(f"Indexing batch {i // batch_size + 1} ({len(batch)} chunks)...")
            db.add_documents(batch)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    
    # Save the index to a specific directory based on the provider
    index_dir = f"faiss_index_{provider}"
    db.save_local(index_dir)
    print(f"FAISS index successfully built and saved to: {index_dir}/")
    
    # Quick verification search
    print("\n--- Verifying Index with Similarity Search ---")
    query = "What is the requirement for minor?"
    docs = db.similarity_search(query, k=2)
    print(f"Query: '{query}'")
    for idx, doc in enumerate(docs):
        print(f"\nMatch {idx+1} (Page {doc.metadata['page']}):")
        print(doc.page_content[:300] + "...")

if __name__ == "__main__":
    # We will build BOTH indexes so that we can easily switch between them in the future!
    print("=== BUILDING LOCAL EMBEDDINGS INDEX ===")
    try:
        build_store(provider="local")
    except Exception as e:
        print("Failed to build local index:", e)
        
    print("\n=== BUILDING GOOGLE EMBEDDINGS INDEX ===")
    try:
        build_store(provider="google")
    except Exception as e:
        print("Failed to build Google index:", e)
