import os
import sqlite3
import re
import requests
import pypdf
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Setup paths
DOWNLOAD_DIR = "scratch/downloaded_plans"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def get_embeddings(provider="local"):
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_file_from_google_drive(file_id, destination):
    """Downloads a public Google Drive file by handling the large file virus warning."""
    api_key = os.getenv("GOOGLE_DRIVE_API_KEY")
    if api_key:
        # Try authenticated Drive API download
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?key={api_key}&alt=media"
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(destination, "wb") as f:
                    for chunk in r.iter_content(32768):
                        if chunk:
                            f.write(chunk)
                return True
            print(f"      Drive API download returned {r.status_code}. Falling back to public link...")
        except Exception as e:
            print(f"      Drive API download failed: {e}. Falling back to public link...")

    # Fallback to public link download
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    try:
        response = session.get(url, params={'id': file_id}, stream=True, timeout=10)
        token = get_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True, timeout=10)
        
        # Save content
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"      Public Drive download failed for ID {file_id}: {e}")
        return False

def download_google_doc_as_pdf(doc_id, destination):
    """Exports a public Google Document as a PDF."""
    api_key = os.getenv("GOOGLE_DRIVE_API_KEY")
    if api_key:
        # Try authenticated Drive API export
        url = f"https://www.googleapis.com/drive/v3/files/{doc_id}/export?key={api_key}&mimeType=application/pdf"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(destination, "wb") as f:
                    f.write(r.content)
                return True
            print(f"      Drive API export returned {r.status_code}. Falling back to public link...")
        except Exception as e:
            print(f"      Drive API export failed: {e}. Falling back to public link...")

    # Fallback to public export
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"
    try:
        r = requests.get(url, allow_redirects=True, timeout=10)
        if r.status_code == 200 and r.headers.get("Content-Type") == "application/pdf":
            with open(destination, "wb") as f:
                f.write(r.content)
            return True
        else:
            print(f"      Doc export failed with status {r.status_code} or wrong Content-Type: {r.headers.get('Content-Type')}")
            return False
    except Exception as e:
        print(f"      Doc export failed for ID {doc_id}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Reads PDF and extracts text."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        print(f"      Error reading PDF {pdf_path}: {e}")
        return None

def build_course_plans_index(provider="local"):
    db_file = "acadagent.db"
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database {db_file} not found. Run ingest_to_sqlite.py first.")
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Query BTech courses with plan URLs
    cursor.execute("""
        SELECT clean_course_code, title, course_plan_url 
        FROM courses 
        WHERE course_plan_url IS NOT NULL;
    """)
    rows = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} course plan URLs in SQLite. Attempting downloads...")
    
    documents = []
    downloaded_count = 0
    
    for code, title, url in rows:
        print(f"  Checking course {code} ({title})...")
        
        # Determine file path for temporary download
        safe_code = code.replace(" ", "_")
        pdf_path = os.path.join(DOWNLOAD_DIR, f"{safe_code}_plan.pdf")
        
        # Regex matches
        drive_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
        docs_match = re.search(r'docs\.google\.com/document/d/([a-zA-Z0-9_-]+)', url)
        
        success = False
        if drive_match:
            file_id = drive_match.group(1)
            print(f"    Downloading Google Drive ID: {file_id}")
            success = download_file_from_google_drive(file_id, pdf_path)
        elif docs_match:
            doc_id = docs_match.group(1)
            print(f"    Exporting Google Doc ID: {doc_id}")
            success = download_google_doc_as_pdf(doc_id, pdf_path)
        else:
            print("    Skipping external or Google Sites link (cannot directly download as PDF).")
            
        if success and os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path)
            if text:
                print(f"    Successfully extracted {len(text)} characters of syllabus text.")
                # Create LangChain document structure
                from langchain_core.documents import Document
                doc = Document(
                    page_content=text,
                    metadata={
                        "course_code": code,
                        "title": title,
                        "url": url,
                        "source": f"{safe_code}_plan.pdf"
                    }
                )
                documents.append(doc)
                downloaded_count += 1
            else:
                print("    Failed to extract text from downloaded PDF.")
                
    if not documents:
        print("No course plan text could be downloaded and extracted. Vector index will not be created.")
        return
        
    print(f"\nSuccessfully downloaded and extracted text for {downloaded_count} course plans.")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split course plans into {len(chunks)} chunks.")
    
    # Build FAISS
    embeddings = get_embeddings(provider)
    print(f"Building FAISS index for course plans using provider '{provider}'...")
    
    if provider == "google":
        # Handle free tier rate limits
        batch_size = 50
        db = FAISS.from_documents(chunks[:batch_size], embeddings)
        for i in range(batch_size, len(chunks), batch_size):
            print("Sleeping 60 seconds to reset rate limits...")
            time.sleep(60)
            batch = chunks[i:i+batch_size]
            print(f"Indexing batch {i // batch_size + 1}... ({len(batch)} chunks)")
            db.add_documents(batch)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        
    index_dir = f"faiss_index_course_plans_{provider}"
    db.save_local(index_dir)
    print(f"Course plans FAISS index saved to: {index_dir}/")
    
    # Simple search verification
    print("\n--- Verifying Course Plan Index Search ---")
    query = "What is taught in computing or programming course?"
    results = db.similarity_search(query, k=2)
    print(f"Query: '{query}'")
    for idx, r in enumerate(results):
        print(f"  Match {idx+1}: {r.metadata['course_code']} ({r.metadata['title']})")
        print(f"  Snippet: {r.page_content[:200]}...")

if __name__ == "__main__":
    # We will build the local provider index first
    print("=== BUILDING LOCAL COURSE PLANS INDEX ===")
    try:
        build_course_plans_index(provider="local")
    except Exception as e:
        print("Local build failed:", e)
        
    # We can skip building Google for now to avoid quota locks since local is fully working
    print("\nLocal index setup complete.")
