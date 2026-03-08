import json
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Import our free local RAG retriever
from src.rag import retrieve_context

load_dotenv()

# PLAN B: Initialize the free Groq LLM instead of OpenAI
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- 1. PARSER AGENT ---
class MathProblem(BaseModel):
    problem_text: str = Field(description="The cleaned math problem.")
    topic: str = Field(description="The core topic.")
    variables: List[str] = Field(description="Variables found (e.g., 'x', 'y').")
    constraints: List[str] = Field(description="Constraints mentioned.")
    needs_clarification: bool = Field(description="True if text is gibberish or missing crucial info.")

def parser_agent(raw_input: str) -> dict:
    print("🕵️ Running Parser Agent...")
    structured_llm = llm.with_structured_output(MathProblem)
    prompt = f"Clean and structure this raw math problem transcription. If it's incomplete or gibberish, flag needs_clarification as true.\n\nInput: {raw_input}"
    try:
        return structured_llm.invoke(prompt).model_dump()
    except Exception as e:
        return {"problem_text": raw_input, "topic": "unknown", "variables": [], "constraints": [], "needs_clarification": True, "error": str(e)}

# --- 2. INTENT ROUTER AGENT ---
class RouteDecision(BaseModel):
    category: str = Field(description="Must be one of: 'algebra', 'probability', 'calculus', 'linear_algebra', 'unknown'")
    requires_rag: bool = Field(description="True if looking up a formula in the knowledge base would help.")

def router_agent(parsed_data: dict) -> dict:
    print("🔀 Running Router Agent...")
    structured_llm = llm.with_structured_output(RouteDecision)
    prompt = f"Classify this math problem and decide if it needs formula retrieval.\nProblem: {parsed_data['problem_text']}"
    try:
        decision = structured_llm.invoke(prompt)
        return decision.model_dump()
    except Exception as e:
         return {"category": "unknown", "requires_rag": False}

# --- 3. SOLVER AGENT ---
def solver_agent(parsed_data: dict, use_rag: bool) -> str:
    print("⚙️ Running Solver Agent...")
    context = ""
    if use_rag:
        print("📚 Solver is fetching RAG context...")
        context = retrieve_context(parsed_data['problem_text'])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert math solver. Solve the problem step-by-step. Use this context if helpful:\n{context}"),
        ("user", "Problem: {problem}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "problem": parsed_data['problem_text']})
    return response.content

# --- 4. VERIFIER AGENT ---
class VerificationResult(BaseModel):
    is_correct: bool = Field(description="True if the solution appears mathematically sound.")
    confidence_score: float = Field(description="Score from 0.0 to 1.0 representing confidence.")
    feedback: str = Field(description="Explanation of any errors found or confirmation of correctness.")

def verifier_agent(problem: str, solution: str) -> dict:
    print("✅ Running Verifier Agent...")
    structured_llm = llm.with_structured_output(VerificationResult)
    prompt = f"Verify this math solution.\nProblem: {problem}\nSolution: {solution}\nCheck for edge cases, units, and logical errors."
    try:
        return structured_llm.invoke(prompt).model_dump()
    except Exception as e:
        return {"is_correct": False, "confidence_score": 0.0, "feedback": "Verification failed."}

# --- 5. EXPLAINER AGENT ---
def explainer_agent(problem: str, solution: str) -> str:
    print("🎓 Running Explainer Agent...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly tutor. Take this raw math solution and explain it to a student in a simple, encouraging way."),
        ("user", "Problem: {problem}\nRaw Solution: {solution}")
    ])
    chain = prompt | llm
    return chain.invoke({"problem": problem, "solution": solution}).content


# --- MASTER PIPELINE FUNCTION ---
def run_full_pipeline(raw_text: str) -> dict:
    """Orchestrates all 5 agents and returns the final state."""
    
    # 1. Parse [cite: 51, 52]
    parsed = parser_agent(raw_text)
    if parsed.get("needs_clarification"):
        return {"status": "hitl_required", "reason": "Parser detected ambiguity.", "parsed": parsed}
        
    # 2. Route [cite: 81, 82]
    route = router_agent(parsed)
    
    # 3. Solve (using RAG) [cite: 83, 84]
    raw_solution = solver_agent(parsed, use_rag=route['requires_rag'])
    
    # 4. Verify [cite: 85, 86]
    verification = verifier_agent(parsed['problem_text'], raw_solution)
    if not verification['is_correct'] or verification['confidence_score'] < 0.8:
         return {
             "status": "hitl_required", 
             "reason": "Verifier lacks confidence or found an error.",
             "parsed": parsed,
             "raw_solution": raw_solution,
             "verification": verification
         }
         
    # 5. Explain [cite: 91, 92]
    final_explanation = explainer_agent(parsed['problem_text'], raw_solution)
    
    return {
        "status": "success",
        "parsed": parsed,
        "route": route,
        "raw_solution": raw_solution,
        "verification": verification,
        "final_explanation": final_explanation
    }

# --- Quick local test ---
if __name__ == "__main__":
    test_problem = "What is the probability of rolling a 6 on a fair six-sided die?"
    print(f"\n--- Starting Pipeline for: {test_problem} ---\n")
    final_result = run_full_pipeline(test_problem)
    print("\n--- Pipeline Complete! ---")
    print(json.dumps(final_result, indent=2))