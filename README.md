# 🧮 AI Math Mentor: Multimodal Agentic Solver

[cite_start]An end-to-end AI application designed to reliably solve JEE-style math problems, explain solutions step-by-step, and continuously improve over time[cite: 4]. [cite_start]This project demonstrates advanced AI engineering patterns including Retrieval-Augmented Generation (RAG), Multi-Agent Orchestration, Human-in-the-Loop (HITL) workflows, and Self-Learning Memory [cite: 2, 6-10].## 🌟 Key Features* [cite_start]**Multimodal Inputs:** Accepts typed text, image uploads (via Tesseract OCR), and audio recordings (via Groq Whisper ASR)[cite: 34, 36, 43, 50].
* [cite_start]**Multi-Agent Architecture:** Utilizes a pipeline of 5 distinct LLM agents (Parser, Router, Solver, Verifier, Explainer) to break down and solve complex problems [cite: 78-92].* [cite_start]**Retrieval-Augmented Generation (RAG):** Grounds the solver agent in a curated mathematical knowledge base using local HuggingFace embeddings and FAISS vector storage.
* [cite_start]**Human-in-the-Loop (HITL):** Automatically pauses execution and requests human intervention if OCR/ASR confidence is low, input is ambiguous, or the verifier agent detects a potential error [cite: 120-125].* [cite_start]**Self-Learning Memory:** Stores human corrections and successful solution patterns in a vector database, allowing the system to reuse known patterns on similar future problems without model retraining [cite: 128-138].

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Input] --> B{Input Type}
    B -->|Text| C[Text Processor]
    B -->|Image| D[Tesseract OCR]
    B -->|Audio| E[Whisper ASR]
    
    C --> F[🕵️ Parser Agent]
    D --> F
    E --> F
    
    F -->|Ambiguous?| G[🛑 HITL: Edit Input]
    G --> F
    
    F -->|Parsed JSON| H[🔀 Intent Router Agent]
    H --> I[⚙️ Solver Agent]
    
    J[(FAISS: Knowledge Base)] -->|RAG Context| I
    K[(FAISS: Memory Store)] -->|Past Patterns| I
    
    I -->|Raw Solution| L[✅ Verifier Agent]
    L -->|Unsure / Error?| M[🛑 HITL: Human Review]
    M --> N[💾 Save to Memory]
    
    L -->|Verified| O[🎓 Explainer Agent]
    O --> P[Streamlit UI]
    
    P --> Q{User Feedback}
    Q -->|Correct| N
    Q -->|Incorrect| M
📁 File Structure & Explanation
app.py: The main Streamlit user interface. It handles user inputs, displays the agent trace, and manages the session state .
src/input_handler.py: Manages the multimodal parsing. It uses pytesseract for image OCR and Groq's API for Whisper audio transcription.
src/agents.py: The core "brain" of the application. It contains the Pydantic models and LangChain logic for the 5 agents (Parser, Router, Solver, Verifier, Explainer) .

src/rag.py: Handles the ingestion, chunking, and embedding of the math knowledge base using HuggingFace and FAISS.

src/memory.py: Manages the self-learning memory layer, saving past interactions to a JSON log and a secondary FAISS database for runtime pattern reuse .
data/: A directory containing the knowledge_base/ text files, the faiss_index/, and the memory_faiss/ databases.

.env.example: A template showing the required environment variables to run the project locally.
🛠️ Tech Stack

Frontend: Streamlit 
LLM Orchestration: LangChain, Groq API (Llama-3.3-70b-versatile)

Embeddings & Vector Store: HuggingFace (all-MiniLM-L6-v2), FAISS 

Multimodal Processing: Tesseract OCR, Groq Whisper 
🚀 Setup & Installation Instructions
1. Clone the repository
Bash

git clone [https://github.com/yourusername/math_mentor.git](https://github.com/yourusername/math_mentor.git)cd math_mentor
2. Set up a virtual environment
Bash

python -m venv venv
venv\Scripts\activate
3. Install Python dependencies
Bash

pip install -r requirements.txt
4. Install Tesseract OCR (Required for Image Input)
Download the 64-bit Windows installer from UB-Mannheim Tesseract OCR. By default, the app expects it to be installed at C:\Program Files\Tesseract-OCR\tesseract.exe.
5. Configure API Keys
Create a .env file in the root directory and add your free Groq API key:
Plaintext

GROQ_API_KEY=gsk_your_actual_api_key_here
6. Initialize the Knowledge Base
Before running the app for the first time, build the local RAG vector database:
Bash

python -m src.rag
7. Run the Application
Launch the Streamlit server:
Bash

streamlit run app.py
Navigate to http://localhost:8501 in your web browser to use the app! 
🔗 Deliverables

Live Application: [Link to your deployed app] 

Demo Video: [Link to your 3-5 minute YouTube demo video] 

---

*Note: Before you push this to GitHub, make sure to replace `yourusername` in the clone link, and eventually fill in the links at the very bottom under "Deliverables".*

You have the code running perfectly and your GitHub repository ready to go. The assi

give this everything in the mark down formate 
for the readme.md 
