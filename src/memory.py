import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

MEMORY_LOG_FILE = "data/memory_log.json"
MEMORY_DB_PATH = "data/memory_faiss"

class MemoryManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        os.makedirs("data", exist_ok=True)
        # Ensure the JSON log exists
        if not os.path.exists(MEMORY_LOG_FILE):
            with open(MEMORY_LOG_FILE, "w") as f:
                json.dump([], f)

    def save_to_memory(self, original_input, parsed, context, raw_solution, verifier_outcome, feedback, is_correct):
        """Saves the interaction to a JSON log and updates the FAISS memory database."""
        # 1. Save to JSON log
        memory_entry = {
            "original_input": original_input,
            "parsed_question": parsed,
            "retrieved_context": context,
            "final_answer": raw_solution,
            "verifier_outcome": verifier_outcome,
            "feedback": feedback,
            "is_correct": is_correct
        }
        
        with open(MEMORY_LOG_FILE, "r") as f:
            memories = json.load(f)
        memories.append(memory_entry)
        with open(MEMORY_LOG_FILE, "w") as f:
            json.dump(memories, f, indent=4)

        # 2. Add to FAISS Vector Store for runtime pattern reuse
        doc = Document(
            page_content=parsed.get("problem_text", original_input),
            metadata={"solution": raw_solution, "feedback": feedback, "is_correct": is_correct}
        )
        
        if os.path.exists(MEMORY_DB_PATH):
            vectorstore = FAISS.load_local(MEMORY_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents([doc])
            vectorstore.save_local(MEMORY_DB_PATH)
        else:
            vectorstore = FAISS.from_documents([doc], self.embeddings)
            vectorstore.save_local(MEMORY_DB_PATH)
            
        print("💾 Interaction saved to Memory!")

    def retrieve_similar_problem(self, query: str):
        """Retrieves a previously solved similar problem to reuse patterns."""
        if not os.path.exists(MEMORY_DB_PATH):
            return None
            
        try:
            vectorstore = FAISS.load_local(MEMORY_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            # Find the single most similar past problem
            docs = vectorstore.similarity_search_with_score(query, k=1)
            
            if docs:
                doc, score = docs[0]
                # Lower score means it is a closer match. 
                if score < 1.0: 
                    print("🧠 Memory Match Found! Reusing past pattern...")
                    return {
                        "past_problem": doc.page_content,
                        "past_solution": doc.metadata.get("solution"),
                        "past_feedback": doc.metadata.get("feedback")
                    }
        except Exception as e:
            print(f"Memory retrieval error: {e}")
            
        return None