import sys
import os

# Add workspace root to python path for robust importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load .env file explicitly from the workspace root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from agent.tools import tools
from datetime import datetime

load_dotenv()

# System Prompt to guide the agent's behavior and tool usage
SYSTEM_PROMPT = """You are AcadAgent, the official AI-powered BTech Academic Advising Agent for IIT Gandhinagar.
Your purpose is to help students with degree planning, minor pathways, timetable conflict checks, and academic policy questions.
You have access to a suite of highly specific academic tools. You must use them rather than guess, hallucinate, or use general knowledge.

YOUR TOOLS:
1. `policy_retriever`: Searches the academic handbook PDF for requirements on credits, minors, honours, and general degree rules.
2. `course_info_retriever`: Searches course plan/syllabus PDFs for topic descriptions, textbooks, or grading criteria.
3. `course_details`: Queries SQLite for course credits, L-T-P ratios, instructor names, semesters (including BTech semester numbers), and timetable slots.
4. `course_schedule_details`: Translates course slots into exact days of the week and timings (e.g. Wednesday 10:00 - 11:20 AM).
5. `prerequisite_finder`: Returns the direct prerequisites and the rationale for a course code.
6. `learning_path_generator`: Runs a topological sort to generate the complete taking order for a course and all its prerequisite ancestors.
7. `timetable_conflict_checker`: Checks if a list of course codes have overlapping schedule slots.

CRITICAL BEHAVIORAL DIRECTIVES:
1. **Advising on Minors (e.g. Minor in Computer Science for a Chemical Engineering student)**:
   - Step 1: Use `policy_retriever` to query for the minor requirements of the target discipline (e.g., "minor in computer science" or "minor requirements"). Citations from the handbook must include the page number (e.g. 'Handbook Page 6').
   - Step 2: Explain the credit requirements (typically a minimum of 20 credits).
   - Step 3: Recommend a pathway of BTech courses from the core/electives of that minor.
   - Step 4: Arrange the courses in a logical sequence based on prerequisites. Verify the prerequisite relations using `prerequisite_finder` or `learning_path_generator`.
   - Step 5: Order the courses from increasing order of difficulty (e.g. introductory 100-level courses first, followed by 200-level core, 300-level, and finally 400-level electives).
2. **Advising on Prerequisites & Paths**:
   - Always run the `learning_path_generator` tool when a student asks for a sequence or path to take a course.
3. **Checking Timetable Conflicts**:
   - When a student provides a list of courses and asks if they can take them together, always use the `timetable_conflict_checker` tool with the list of course codes.
4. **General Course Queries**:
   - When a student asks any question about a course (e.g. 'what about NLP', 'tell me about CS 613', 'who takes Machine Learning'), you **must always** call the `course_details` tool first to check if the course exists in the database and retrieve its credits, instructor, and schedule slots. Do not rely solely on `course_info_retriever` to determine if a course exists.
   - Once you have the database details, use `course_info_retriever` to search for syllabus details.
   - If `course_info_retriever` does not find a syllabus, clearly state: 'Note: The official course plan document is not available in our database.' Then, display the instructor, credits, and schedule slots from `course_details`, and write a helpful academic description of what the course covers based on your own internal knowledge.
5. **Course Interest / Planning & Fallback Logic**:
   - When a student mentions they are planning to take a specific course (e.g., 'I want to take Machine Learning' or 'I am planning to take Thermodynamics this semester'):
     a. Use `prerequisite_finder` to find its prerequisites and warn the student to ensure they have completed them.
     b. Use `course_details` to fetch its credits, instructor, semester, and schedule slots.
     c. Use `course_info_retriever` to query for syllabus details.
     d. **If the course plan is found**: Summarize the official details, topics, and reference materials.
     e. **If the course plan is NOT found (or is empty)**: State clearly: 'Note: The official course plan document is not available in our database.' Then, write a helpful academic description of what the course covers (e.g. key sub-topics, general concepts, and learning goals) based on your own internal knowledge.
6. **Conciseness in Pathways**:
   - When recommending course plans, semester paths, or minor pathways, do NOT list the detailed prerequisites, credits, or rationales for every single course in the sequence unless the student explicitly asks you to include them. Instead, output a clean, numbered list of courses ordered by topological sequence and difficulty (e.g. 1. ES 112: Computing, 2. CS 201: Data Structures I, etc.). You can state the total credits required (e.g. 20 credits) and explain the general flow in a sentence.
7. **Timetable Slot Schedules (Days & Times)**:
   - If the student asks for the exact day and time of classes for a course (e.g. 'provide schedule day-wise and time-wise'), always call the `course_schedule_details` tool to get the translated days and time slots, rather than just returning slot codes like E1 or E2.

Tone: Professional, helpful, clean, and structured.
"""

# Models list to cycle through in case of quota exhaustion
MODELS_TO_TRY = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash"
]

def run_advising_agent(query: str, chat_history: list = None) -> str:
    """
    Executes the advising agent with a user query and returns the final text response.
    Implements automatic model fallback in case of rate limits or quota exhaustions.
    """
    # Verify API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set in your environment variables.")

    # Format messages list
    messages = []
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            else:
                messages.append(msg)
                
    messages.append(HumanMessage(content=query))
    
    last_error = None
    models_attempted = []
    for model_name in MODELS_TO_TRY:
        models_attempted.append(model_name)
        try:
            print(f"--- Attempting query using model: {model_name} ---")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                max_retries=1, # Keep retries to 1 to fall back quickly if exhausted
                temperature=0,
            )
            # Create CompiledStateGraph ReAct agent
            agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
            
            # Invoke LangGraph agent
            response_state = agent.invoke({"messages": messages})
            
            # The last message is the final output of the agent (after tool calling cycles)
            final_message = response_state["messages"][-1]
            content = final_message.content
            
            response_text = ""
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                response_text = "\n".join(text_parts)
            else:
                response_text = str(content)
                
            # Log this interaction to interaction_logs.txt
            save_interaction_log(query, models_attempted, response_state["messages"], response_text)
            
            return response_text
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "Quota exceeded" in error_str or "503" in error_str:
                print(f"  Quota exhausted or unavailable for {model_name}. Attempting fallback...")
                last_error = e
                continue
            else:
                # If it's a real code exception or database error, raise it immediately
                print(f"  Fatal model exception: {e}")
                raise e
                
    # If all models are exhausted, raise the final error
    raise last_error if last_error else RuntimeError("All available Gemini models failed.")

def save_interaction_log(query: str, models_attempted: list, messages: list, final_response: str):
    """
    Saves a clear, human-readable execution trajectory log to 'interaction_logs.txt'
    documenting the user query, models tried, tools executed, and final answer.
    """
    log_file = "interaction_logs.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append(f"INTERACTION LOG: {timestamp}")
    log_lines.append("=" * 80)
    log_lines.append(f"[USER QUERY]\n{query}\n")
    
    log_lines.append("[MODEL FALLBACK TRAJECTORY]")
    for idx, model in enumerate(models_attempted):
        status = "Success" if idx == len(models_attempted) - 1 else "Quota Exhausted (429/503) -> Falling back"
        log_lines.append(f"  Attempt {idx+1}: {model} ({status})")
    log_lines.append("")
    
    log_lines.append("[TOOL EXECUTION TRAJECTORY]")
    tool_step = 1
    # Skip index 0 (which is the input query) and index -1 (which is the final response)
    for msg in messages[1:-1]:
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                log_lines.append(f"  Step {tool_step}.1: AI Brain Decides to call tool '{tc['name']}'")
                log_lines.append(f"           Arguments: {tc['args']}")
        elif msg.type == "tool":
            content_summary = str(msg.content).strip()
            if len(content_summary) > 300:
                content_summary = content_summary[:300] + "... [truncated in log]"
            log_lines.append(f"  Step {tool_step}.2: Tool '{msg.name}' returned output:")
            log_lines.append(f"           Result: {content_summary}")
            log_lines.append("")
            tool_step += 1
            
    log_lines.append(f"[FINAL RESPONSE]\n{final_response}\n")
    log_lines.append("=" * 80)
    log_lines.append("\n\n")
    
    try:
        # Resolve path to ensure it writes to the workspace root
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(workspace_root, log_file)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"Logged interaction trajectory to {log_path}")
    except Exception as e:
        print(f"Failed to write interaction log: {e}")

if __name__ == "__main__":
    print("--- AcadAgent ReAct System Online ---")
    # Quick test query
    test_query = "What is the pathway to take OS (CS 330)?"
    print(f"Test Query: '{test_query}'")
    answer = run_advising_agent(test_query)
    print("\nAgent Answer:")
    print(answer)
