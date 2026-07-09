import os
import pickle
import sqlite3
import re
import networkx as nx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env file explicitly from the workspace root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# Determine embedding provider (toggles between 'local' and 'google')
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# Global variables to cache loaders
_embeddings_cache = None
_policy_db_cache = None
_plans_db_cache = None

def get_embeddings_model():
    global _embeddings_cache
    if _embeddings_cache is None:
        if EMBEDDING_PROVIDER == "google":
            _embeddings_cache = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        else:
            _embeddings_cache = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings_cache

def load_policy_index():
    global _policy_db_cache
    if _policy_db_cache is None:
        index_dir = f"faiss_index_{EMBEDDING_PROVIDER}"
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"FAISS index folder '{index_dir}' not found. Run build_vector_store.py first.")
        _policy_db_cache = FAISS.load_local(index_dir, get_embeddings_model(), allow_dangerous_deserialization=True)
    return _policy_db_cache

def load_plans_index():
    global _plans_db_cache
    if _plans_db_cache is None:
        index_dir = f"faiss_index_course_plans_{EMBEDDING_PROVIDER}"
        # Fallback to local if google wasn't built (due to quota limits)
        if not os.path.exists(index_dir) and EMBEDDING_PROVIDER == "google":
            index_dir = "faiss_index_course_plans_local"
        if not os.path.exists(index_dir):
            # If still not found, return None (RAG over plans will be skipped gracefully)
            return None
        _plans_db_cache = FAISS.load_local(index_dir, get_embeddings_model(), allow_dangerous_deserialization=True)
    return _plans_db_cache

def clean_code(raw_code):
    if not raw_code:
        return ""
    m = re.search(r'([A-Za-z]{2,3})\s*(\d{3})', raw_code)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    return raw_code.strip().upper()

COMMON_ABBREVIATIONS = {
    "nlp": "Natural Language Processing",
    "ml": "Machine Learning",
    "os": "Operating Systems",
    "cn": "Computer Networks",
    "ai": "Artificial Intelligence",
    "dsa": "Data Structures",
    "oop": "Object Oriented Programming",
    "dl": "Deep Learning",
    "dbms": "Database Systems",
    "db": "Database Systems"
}

def resolve_course_code(input_str: str) -> str:
    """
    Attempts to resolve an input string (either a course code or a natural language title)
    to the official clean course code format (e.g. 'CS 330' or 'ES 335') using SQLite.
    """
    if not input_str:
        return ""
        
    # Map common academic shorthand to full titles
    lookup_term = input_str.strip().lower()
    if lookup_term in COMMON_ABBREVIATIONS:
        input_str = COMMON_ABBREVIATIONS[lookup_term]
        
    cleaned = clean_code(input_str)
    
    # Check if the cleaned format matches a standard course code (e.g., CS 330, ES 112)
    if re.match(r'^[A-Z]{2,3}\s*\d{3}$', cleaned):
        return cleaned
        
    # If not a standard code, treat it as a title and search SQLite
    db_file = "acadagent.db"
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # 1. Try exact title match (case-insensitive)
            cursor.execute("SELECT clean_course_code FROM courses WHERE LOWER(title) = ? LIMIT 1;", (input_str.strip().lower(),))
            row = cursor.fetchone()
            if row:
                conn.close()
                return row[0]
                
            # 2. Try partial match (LIKE search)
            search_pattern = f"%{input_str.strip().lower()}%"
            cursor.execute("SELECT clean_course_code FROM courses WHERE LOWER(title) LIKE ? LIMIT 1;", (search_pattern,))
            row = cursor.fetchone()
            if row:
                conn.close()
                return row[0]
                
            conn.close()
        except Exception as e:
            print(f"Error resolving course code from title '{input_str}': {e}")
            
    # Fallback to the cleaned string
    return cleaned

# --- TOOL 1: PREREQUISITE FINDER ---
@tool
def prerequisite_finder(course_code: str) -> str:
    """
    Finds the direct prerequisites for a given course code.
    Input: Course code, e.g., 'CS 201' or 'ES 242'
    """
    graph_file = "graph/prerequisite_graph.pkl"
    if not os.path.exists(graph_file):
        return "Prerequisite graph not found. Run build_prerequisite_graph.py first."
        
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
        
    cleaned = resolve_course_code(course_code)
    if not G.has_node(cleaned):
        return f"Course '{course_code}' (cleaned: '{cleaned}') was not found in the prerequisite graph."
        
    prereqs = list(G.predecessors(cleaned))
    if not prereqs:
        return f"Course '{cleaned}' has no direct prerequisites listed in the graph."
        
    result = [f"Direct prerequisites for {cleaned}:"]
    for pre in prereqs:
        # Get rationale stored on the edge
        edge_data = G.get_edge_data(pre, cleaned)
        rationale = edge_data.get("rationale", "No reason provided.")
        title = G.nodes[pre].get("title", "Unknown Title")
        result.append(f"  - {pre} ({title}): {rationale}")
        
    return "\n".join(result)

# --- TOOL 2: LEARNING PATH GENERATOR ---
@tool
def learning_path_generator(target_course: str) -> str:
    """
    Generates a valid, ordered sequence of courses (topological sort) 
    that must be completed to take a specific target course.
    Input: Target course code, e.g., 'CS 330' or 'ES 335'
    """
    graph_file = "graph/prerequisite_graph.pkl"
    if not os.path.exists(graph_file):
        return "Prerequisite graph not found. Run build_prerequisite_graph.py first."
        
    with open(graph_file, "rb") as f:
        G = pickle.load(f)
        
    cleaned = resolve_course_code(target_course)
    if not G.has_node(cleaned):
        return f"Course '{target_course}' (cleaned: '{cleaned}') was not found in the prerequisite graph."
        
    # Get all ancestors (recursive prerequisites)
    try:
        ancestors = nx.ancestors(G, cleaned)
    except Exception as e:
        return f"Error resolving prerequisites: {e}"
        
    if not ancestors:
        return f"Course '{cleaned}' has no prerequisite dependencies. You can take it immediately!"
        
    # Build subgraph of target course and its ancestors
    nodes_to_sort = list(ancestors) + [cleaned]
    subgraph = G.subgraph(nodes_to_sort)
    
    # Run topological sort
    try:
        ordered_sequence = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        return f"Error: Circular dependency detected in the prerequisite graph for {cleaned}!"
        
    result = [f"To take {cleaned} ({G.nodes[cleaned].get('title', 'Unknown')}), you should follow this pathway:"]
    for idx, course in enumerate(ordered_sequence):
        title = G.nodes[course].get("title", "Unknown Title")
        if course == cleaned:
            result.append(f"  {idx+1}. [TARGET] {course}: {title}")
        else:
            result.append(f"  {idx+1}. {course}: {title}")
            
    return "\n".join(result)

# --- TOOL 3: COURSE DETAILS ---
@tool
def course_details(course_code: str) -> str:
    """
    Fetches official details about a course from the SQLite database, 
    including credits, instructor, semester, BTech semester number, timetable slots, and course plan links.
    Input: Course code, e.g., 'CS 201' or 'MA 104'
    """
    db_file = "acadagent.db"
    if not os.path.exists(db_file):
        return "Courses database not found. Run ingest_to_sqlite.py first."
        
    cleaned = resolve_course_code(course_code)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Query details including semester_num
    cursor.execute("""
        SELECT course_code, title, instructor, semester, semester_num, L, T, P, C, 
               course_plan_url, lecture_slots, tutorial_slots, lab_slots, hss_bs_elective
        FROM courses 
        WHERE clean_course_code = ? OR course_code = ?;
    """, (cleaned, course_code.strip().upper()))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return f"No course details found for '{course_code}' in the database."
        
    result = []
    for idx, row in enumerate(rows):
        code, title, inst, sem, sem_num, l, t, p, c, url, lec, tut, lab, hss = row
        result.append(f"Course Match {idx+1}: {code} - {title}")
        result.append(f"  Instructor: {inst if inst else 'TBD'}")
        result.append(f"  Credits: {c} (L-T-P: {l}-{t}-{p})")
        
        # Display BTech semester number if available
        sem_str = sem.capitalize()
        if sem_num:
            sem_str += f" (Recommended BTech Semester: {sem_num})"
        result.append(f"  Semester: {sem_str}")
        
        # Schedule Slots
        slots = []
        if lec: slots.append(f"Lecture: {lec}")
        if tut: slots.append(f"Tutorial: {tut}")
        if lab: slots.append(f"Lab: {lab}")
        result.append(f"  Timetable Slots: {', '.join(slots) if slots else 'Not scheduled'}")
        
        if url:
            result.append(f"  Course Plan: {url}")
        if hss:
            result.append(f"  Elective Category: {hss}")
        result.append("")
        
    return "\n".join(result)

# Slot timings mapped from the "Time Slots" sheet
SLOT_TIMINGS = {
    'A1': 'Monday 8:30 - 9:50',
    'B1': 'Tuesday 8:30 - 9:50',
    'A2': 'Wednesday 8:30 - 9:50',
    'C2': 'Thursday 8:30 - 9:50',
    'B2': 'Friday 8:30 - 9:50',
    'C1': 'Monday 10:00 - 11:20',
    'D1': 'Tuesday 10:00 - 11:20',
    'E1': 'Wednesday 10:00 - 11:20',
    'D2': 'Thursday 10:00 - 11:20',
    'E2': 'Friday 10:00 - 11:20',
    'F1': 'Monday 11:30 - 12:50',
    'G1': 'Tuesday 11:30 - 12:50',
    'H2': 'Wednesday 11:30 - 12:50',
    'F2': 'Thursday 11:30 - 12:50',
    'G2': 'Friday 11:30 - 12:50',
    'I1': 'Monday 14:00 - 15:20',
    'J1': 'Tuesday 14:00 - 15:20',
    'I2': 'Wednesday 14:00 - 15:20',
    'K2': 'Thursday 14:00 - 15:20',
    'J2': 'Friday 14:00 - 15:20',
    'K1': 'Monday 15:30 - 16:50',
    'L1': 'Tuesday 15:30 - 16:50',
    'M1': 'Wednesday 15:30 - 16:50',
    'L2': 'Thursday 15:30 - 16:50',
    'M2': 'Friday 15:30 - 16:50',
    'H1': 'Monday 17:00 - 18:20',
    'N1': 'Tuesday 17:00 - 18:20',
    'P1': 'Wednesday 17:00 - 18:20',
    'N2': 'Thursday 17:00 - 18:20',
    'P2': 'Friday 17:00 - 18:20',
}

@tool
def course_schedule_details(course_code: str) -> str:
    """
    Returns the day-wise and time-wise schedule (days of the week and timings) for a course's classes.
    Input: Course code or title, e.g. 'CS 330' or 'Operating Systems'
    """
    db_file = "acadagent.db"
    if not os.path.exists(db_file):
        return "Courses database not found. Run ingest_to_sqlite.py first."
        
    cleaned = resolve_course_code(course_code)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT course_code, title, semester, lecture_slots, tutorial_slots, lab_slots
        FROM courses 
        WHERE clean_course_code = ? OR course_code = ?;
    """, (cleaned, course_code.strip().upper()))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return f"No timetable slots found for '{course_code}' in the database."
        
    result = []
    for idx, row in enumerate(rows):
        code, title, sem, lec, tut, lab = row
        result.append(f"Schedule for {code} - {title} ({sem.capitalize()} Semester):")
        
        has_slots = False
        
        # Translate lectures
        if lec:
            has_slots = True
            result.append("  * Lectures:")
            for slot in lec.split(","):
                slot = slot.strip().upper()
                time_info = SLOT_TIMINGS.get(slot, "Custom schedule/Check timetable")
                result.append(f"    - Slot {slot}: {time_info}")
                
        # Translate tutorials
        if tut:
            has_slots = True
            result.append("  * Tutorials:")
            for slot in tut.split(","):
                slot = slot.strip().upper()
                time_info = SLOT_TIMINGS.get(slot, "Custom schedule/Check timetable")
                result.append(f"    - Slot {slot}: {time_info}")
                
        # Translate labs
        if lab:
            has_slots = True
            result.append("  * Labs:")
            for slot in lab.split(","):
                slot = slot.strip().upper()
                time_info = SLOT_TIMINGS.get(slot, "Custom schedule/Check timetable")
                result.append(f"    - Slot {slot}: {time_info}")
                
        if not has_slots:
            result.append("  - No specific slots are assigned to this course.")
        result.append("")
        
    return "\n".join(result)

# --- TOOL 4: POLICY RETRIEVER (RAG over Handbook) ---
@tool
def policy_retriever(query: str) -> str:
    """
    Searches the academic handbook vector index to answer questions about 
    academic rules, credits, requirements, minors, or honours degrees.
    Input: Question in natural language, e.g., 'What are the rules for completing a minor?'
    """
    try:
        db = load_policy_index()
    except Exception as e:
        return f"Error loading policy index: {e}"
        
    docs = db.similarity_search(query, k=3)
    result = ["Relevant Policy Sections found in Academic Handbook:"]
    for idx, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown Page")
        src = doc.metadata.get("source", "handbook.pdf")
        result.append(f"\nSource: {src} (Page {page})")
        result.append("-" * 40)
        result.append(doc.page_content.strip())
        
    return "\n".join(result)

# --- TOOL 5: COURSE INFO RETRIEVER (RAG over Course Plans) ---
@tool
def course_info_retriever(query: str) -> str:
    """
    Searches the course syllabus and course plan documents to answer questions 
    about specific topics, textbooks, grading, or descriptions in a course.
    Input: Query, e.g., 'What topics are covered in computing or data structures?'
    """
    try:
        db = load_plans_index()
    except Exception as e:
        return f"Error loading course plans index: {e}"
        
    if db is None:
        return "Course plans vector index was not built or has no documents indexed."
        
    docs = db.similarity_search(query, k=3)
    result = ["Relevant Syllabus Info found in Course Plans:"]
    for idx, doc in enumerate(docs):
        code = doc.metadata.get("course_code", "Unknown")
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "No Link")
        result.append(f"\nCourse: {code} - {title} (URL: {url})")
        result.append("-" * 40)
        result.append(doc.page_content.strip())
        
    return "\n".join(result)

# --- TOOL 6: TIMETABLE CONFLICT CHECKER ---
@tool
def timetable_conflict_checker(courses_list_str: str) -> str:
    """
    Checks if a list of course codes have overlapping schedule slots.
    Input: Comma-separated course codes, e.g., 'MA 104, ES 242, CS 203'
    """
    codes = [resolve_course_code(c.strip()) for c in courses_list_str.split(",") if c.strip()]
    if len(codes) < 2:
        return "Please provide at least two courses to check for conflicts."
        
    db_file = "acadagent.db"
    if not os.path.exists(db_file):
        return "Courses database not found. Run ingest_to_sqlite.py first."
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    course_slots = {}
    
    for c in codes:
        cursor.execute("""
            SELECT course_code, semester, lecture_slots, tutorial_slots, lab_slots 
            FROM courses 
            WHERE clean_course_code = ? OR course_code = ?;
        """, (c, c))
        rows = cursor.fetchall()
        if not rows:
            course_slots[c] = []
            continue
            
        # Collect slots (handling multiple entries for same code, e.g., odd/even)
        slots_set = set()
        for row in rows:
            # We look at slots. E.g. lecture: 'E1,E2', tutorial: 'H2'
            _, sem, lec, tut, lab = row
            for s_str in [lec, tut, lab]:
                if s_str:
                    for slot in s_str.split(","):
                        # Attach semester to avoid false clashing between an odd-sem course and even-sem course
                        slots_set.add(f"{sem.lower()}_{slot.strip().upper()}")
        course_slots[c] = list(slots_set)
        
    conn.close()
    
    # Check for clashes
    conflicts = []
    # Compare every pair
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            c1 = codes[i]
            c2 = codes[j]
            s1 = course_slots[c1]
            s2 = course_slots[c2]
            
            # Find intersection
            clashes = set(s1).intersection(set(s2))
            if clashes:
                # Format clashes (remove sem prefix)
                clashes_clean = [c.split("_")[1] for c in clashes]
                sem_name = clashes.pop().split("_")[0].capitalize()
                conflicts.append(f"  * Conflict between '{c1}' and '{c2}' in {sem_name} Semester on Slot(s): {', '.join(clashes_clean)}")
                
    if conflicts:
        return "Timetable Conflict Detected!\n" + "\n".join(conflicts)
    else:
        return f"No timetable conflicts detected among the courses: {', '.join(codes)}."

# Export list of tools
tools = [
    prerequisite_finder,
    learning_path_generator,
    course_details,
    course_schedule_details,
    policy_retriever,
    course_info_retriever,
    timetable_conflict_checker
]
