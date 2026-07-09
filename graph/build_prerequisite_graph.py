import os
import sqlite3
import pickle
import networkx as nx
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import List

load_dotenv()

# Define Pydantic Schema for Structured Output
class PrerequisiteRelation(BaseModel):
    prerequisite: str  # E.g., "ES 112"
    course: str        # E.g., "CS 201"
    rationale: str     # E.g., "CS 201 (Data Structures I) requires programming basics from ES 112."

class PrerequisiteList(BaseModel):
    relations: List[PrerequisiteRelation]

def get_courses_by_prefix(cursor, prefix):
    """Fetches all courses matching the prefix from the SQLite database."""
    cursor.execute("""
        SELECT clean_course_code, title 
        FROM courses 
        WHERE clean_course_code LIKE ?
        GROUP BY clean_course_code;
    """, (f"{prefix} %",))
    return cursor.fetchall()

def build_graph():
    db_file = "acadagent.db"
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database {db_file} not found. Run ingest_to_sqlite.py first.")
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Initialize the modern Gemini Client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from the environment.")
    client = genai.Client(api_key=api_key)
    
    # Define prefixes to analyze
    prefixes = ['CS', 'EE', 'CL', 'ME', 'CE', 'MSE', 'BE', 'MA', 'ES']
    
    # First-year foundational courses that can be prerequisites for anything
    foundational_courses = [
        "FP 100 (Foundation Programme)",
        "ES 112 (Computing)",
        "MA 103 (Calculus of Single Variable & Linear Algebra)",
        "MA 104 (Ordinary Differential Equations)",
        "ES 114 (Probability, Statistics & Data Visualization)",
        "ES 116 (Principles & Applications of Electrical Engineering)",
        "ES 117 (The World of Engineering)",
        "ES 119 (Principles of Artificial Intelligence)",
        "PH 101 (Physics)",
        "CH 101 (General Chemistry)",
        "BS 192 (Undergraduate Science Laboratory)"
    ]
    foundational_str = "\n".join([f"- {c}" for c in foundational_courses])
    
    # Initialize NetworkX directed graph
    G = nx.DiGraph()
    
    # Add all unique courses in the DB as nodes first
    cursor.execute("SELECT DISTINCT clean_course_code, title FROM courses;")
    all_db_courses = cursor.fetchall()
    for code, title in all_db_courses:
        G.add_node(code, title=title)
        
    print(f"Added {G.number_of_nodes()} courses as graph nodes.")
    
    # Track identified edges
    all_relations = []
    
    import time
    for prefix in prefixes:
        # Sleep 5 seconds to respect Gemini API rate limits (15 RPM)
        time.sleep(5)
        courses = get_courses_by_prefix(cursor, prefix)
        if not courses:
            continue
            
        print(f"\nAnalyzing department prefix '{prefix}' ({len(courses)} courses)...")
        
        # Format list of courses for the prompt
        courses_str = "\n".join([f"- {code}: {title}" for code, title in courses])
        
        prompt = f"""
You are an academic curriculum planning expert.
Below is a list of courses offered by the '{prefix}' department at IIT Gandhinagar:
{courses_str}

Below are common first-year foundational courses that might be prerequisites for '{prefix}' courses:
{foundational_str}

YOUR TASK:
Identify any direct academic prerequisite relationships among the '{prefix}' courses, or between the foundational courses and the '{prefix}' courses.
A prerequisite means course A MUST be completed before taking course B.

CRITICAL RULES:
1. Be conservative. Only suggest a prerequisite if there is a strong academic dependency. For example:
   - "Data Structures & Algorithms II" requires "Data Structures & Algorithms I".
   - "Operating Systems" or "Computer Networks" requires "Computing" or "Computer Organization".
   - "Advanced Algorithms" requires "Data Structures & Algorithms II" or "Algorithms I".
   - Higher-level math courses require calculus foundations.
2. Only output course codes that exist in the provided lists.
3. If there are no prerequisites for a course, do not add any relation for it.
4. Output should strictly follow the JSON schema.
"""
        # Retry loop for model request to handle temporary 503/429 errors
        success = False
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PrerequisiteList,
                        temperature=0.0
                    )
                )
                success = True
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed for prefix '{prefix}': {e}. Retrying in 10s...")
                time.sleep(10)
                
        if not success or response is None:
            print(f"Error analyzing prefix '{prefix}' after 3 attempts. Skipping.")
            continue
            
        try:
            # Since response_schema is used, LangChain / Gemini returns structured JSON
            import json
            data = json.loads(response.text)
            
            relations = data.get("relations", [])
            print(f"Gemini identified {len(relations)} prerequisite edges for prefix '{prefix}'.")
            
            for rel in relations:
                pre = rel["prerequisite"].strip().upper()
                course = rel["course"].strip().upper()
                rat = rel["rationale"]
                
                # Verify that both course codes exist in our graph nodes to avoid hallucinations
                if G.has_node(pre) and G.has_node(course):
                    G.add_edge(pre, course, rationale=rat)
                    all_relations.append((pre, course, rat))
                    print(f"  Added Edge: {pre} -> {course} | Reason: {rat}")
                else:
                    print(f"  Skipped hallucinated edge: {pre} -> {course}")
                    
        except Exception as e:
            print(f"Error analyzing prefix '{prefix}':", e)
            
    conn.close()
    
    # Save the graph
    os.makedirs("graph", exist_ok=True)
    graph_file = "graph/prerequisite_graph.pkl"
    with open(graph_file, "wb") as f:
        pickle.dump(G, f)
        
    print(f"\n--- Prerequisite Graph Summary ---")
    print(f"Saved graph to {graph_file}")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    # Show a few sample pathways
    print("\nSample edges in the graph:")
    edges_list = list(G.edges(data=True))[:10]
    for u, v, d in edges_list:
        print(f"  {u} -> {v} (Reason: {d.get('rationale')})")

if __name__ == "__main__":
    build_graph()
