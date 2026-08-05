import os
import re
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun

# -----------------------
# Config
# -----------------------
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")

# Choose a reliable, available model for your account
# Options that are commonly available: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Hard limits to prevent infinite loops
MAX_RESEARCH_ROUNDS = 12    # total sub-question research steps across the run
MAX_REFINEMENTS = 2         # refine -> evaluate loops

# -----------------------
# Setup
# -----------------------
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
search_tool = DuckDuckGoSearchRun()

class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    research_findings: Dict[str, str]
    final_answer: Optional[str]
    evaluation: Optional[str]
    refinement_count: int
    research_rounds: int

# -----------------------
# Helpers
# -----------------------
def parse_sub_questions(text: str) -> List[str]:
    # Split by lines, strip bullets/numbers like "1. ", "- ", "* "
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned: List[str] = []
    for l in lines:
        l = re.sub(r"^(\d+[\.\)]\s+|[-*]\s+)", "", l)
        if len(l) > 0:
            cleaned.append(l)
    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for q in cleaned:
        if q not in seen:
            seen.add(q)
            result.append(q)
    return result[:5]  # cap to 3-5; we’ll take up to 5 just in case

def pick_next_unanswered(sub_questions: List[str], findings: Dict[str, str]) -> Optional[str]:
    for q in sub_questions:
        if q not in findings:
            return q
    return None

# -----------------------
# Nodes
# -----------------------
def breakdown_question(state: AgentState) -> AgentState:
    print("[Node] breakdown_question")
    messages = [
        SystemMessage(content="Break the main question into 3-5 concise, standalone sub-questions. Return each on a new line without bullets or numbers."),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(messages)
    sub_questions = parse_sub_questions(response.content)
    if not sub_questions:
        # Fallback to the main question if the model didn't produce clear sub-questions
        sub_questions = [state["question"]]
    return {
        **state,
        "sub_questions": sub_questions,
        "research_findings": {},
        "refinement_count": 0,
        "research_rounds": 0,
    }

def research_with_search(state: AgentState) -> AgentState:
    print("[Node] research_with_search")
    # Select the next unanswered sub-question here (do not mutate state in conditional edges)
    next_q = pick_next_unanswered(state["sub_questions"], state["research_findings"])
    if next_q is None:
        # Nothing left to research
        return state

    # Hard stop guard for research loops
    if state["research_rounds"] >= MAX_RESEARCH_ROUNDS:
        print("  [Guard] MAX_RESEARCH_ROUNDS reached; skipping further research.")
        return state

    try:
        search_results = search_tool.invoke(next_q)
    except Exception as e:
        search_results = f"(Search error) {e}"

    messages = [
        SystemMessage(content="Answer the sub-question using the provided web search snippets. Be concise, cite key points, and avoid speculation."),
        HumanMessage(content=f"Sub-question: {next_q}\n\nSearch snippets:\n{search_results[:3000]}")
    ]
    response = llm.invoke(messages)

    findings = dict(state["research_findings"])
    findings[next_q] = response.content

    return {
        **state,
        "research_findings": findings,
        "research_rounds": state["research_rounds"] + 1,
    }

def select_next_step(state: AgentState) -> str:
    # Do not mutate state here. Only decide the route.
    remaining = [q for q in state["sub_questions"] if q not in state["research_findings"]]
    if remaining and state["research_rounds"] < MAX_RESEARCH_ROUNDS:
        return "research"
    return "synthesize"

def synthesize_findings(state: AgentState) -> AgentState:
    print("[Node] synthesize_findings")
    findings_text = "\n\n".join([f"{q}:\n{a}" for q, a in state["research_findings"].items()])
    messages = [
        SystemMessage(content="Synthesize a clear, structured answer to the main question using the findings provided. Be balanced, specific, and concise."),
        HumanMessage(content=f"Main question:\n{state['question']}\n\nFindings:\n{findings_text if findings_text else '(No findings gathered)'}")
    ]
    response = llm.invoke(messages)
    return {**state, "final_answer": response.content}

def evaluate_answer(state: AgentState) -> AgentState:
    print("[Node] evaluate_answer")
    messages = [
        SystemMessage(content="Evaluate the answer for correctness, completeness, and clarity. Reply with ONLY one word: excellent, adequate, or needs improvement."),
        HumanMessage(content=f"Answer:\n{state.get('final_answer') or '(empty)'}")
    ]
    response = llm.invoke(messages)
    label = (response.content or "").strip().lower()
    # Normalize to the three labels in case the model adds extra text
    if "excellent" in label:
        label = "excellent"
    elif "adequate" in label:
        label = "adequate"
    else:
        label = "needs improvement"
    return {**state, "evaluation": label}

def needs_refinement(state: AgentState) -> str:
    # Hard stop after MAX_REFINEMENTS
    if state["refinement_count"] >= MAX_REFINEMENTS:
        return "complete"
    if state["evaluation"] in ("excellent", "adequate"):
        return "complete"
    return "refine"

def refine_answer(state: AgentState) -> AgentState:
    print("[Node] refine_answer")
    messages = [
        SystemMessage(content="Refine the answer to address weaknesses and improve clarity, structure, and specificity."),
        HumanMessage(content=f"Main question:\n{state['question']}\n\nCurrent answer:\n{state.get('final_answer') or '(empty)'}\n\nEvaluation: {state.get('evaluation')}")
    ]
    response = llm.invoke(messages)
    return {**state, "final_answer": response.content, "refinement_count": state["refinement_count"] + 1}

# -----------------------
# Graph
# -----------------------
graph = StateGraph(AgentState)
graph.add_node("breakdown", breakdown_question)
graph.add_node("research", research_with_search)
graph.add_node("synthesize", synthesize_findings)
graph.add_node("evaluate", evaluate_answer)
graph.add_node("refine", refine_answer)

graph.add_edge("breakdown", "research")
graph.add_conditional_edges("research", select_next_step, {"research": "research", "synthesize": "synthesize"})
graph.add_edge("synthesize", "evaluate")
graph.add_conditional_edges("evaluate", needs_refinement, {"refine": "refine", "complete": END})
graph.add_edge("refine", "evaluate")

graph.set_entry_point("breakdown")
research_agent = graph.compile()

# -----------------------
# Runner
# -----------------------
def run_research_agent(question: str):
    initial: AgentState = {
        "question": question,
        "sub_questions": [],
        "research_findings": {},
        "final_answer": None,
        "evaluation": None,
        "refinement_count": 0,
        "research_rounds": 0,
    }
    final_state = research_agent.invoke(initial)
    print("\n=== Final Answer ===\n", final_state.get("final_answer"))
    print("\n=== Evaluation ===\n", final_state.get("evaluation"))
    print("\n=== Stats ===")
    print("  Sub-questions:", len(final_state.get("sub_questions", [])))
    print("  Findings:", len(final_state.get("research_findings", {})))
    print("  Research rounds:", final_state.get("research_rounds"))
    print("  Refinements:", final_state.get("refinement_count"))

if __name__ == "__main__":
    run_research_agent("List all e-mail address linked to ROBSON TAVARES NONATO?")