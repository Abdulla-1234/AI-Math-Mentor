import streamlit as st
import os
import tempfile
from src.input_handler import process_text, process_image, process_audio
from src.agents import run_full_pipeline
from src.memory import MemoryManager

# Configure the Streamlit page
st.set_page_config(page_title="AI Math Mentor", page_icon="🧮", layout="wide")

st.title("🧮 Reliable Multimodal Math Mentor")
st.markdown("Upload a photo, record audio, or type your JEE-style math problem! [cite: 14-17]")

# --- INITIALIZE MEMORY ---
# Use cache_resource so it doesn't rebuild the memory manager on every click
@st.cache_resource
def get_memory_manager():
    return MemoryManager()
memory_manager = get_memory_manager()

# --- INITIALIZE SESSION STATE ---
# This prevents data from disappearing when buttons are clicked
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 1.0

# --- 1. INPUT MODE SELECTOR ---
input_mode = st.radio("Choose Input Mode:", ["Text", "Image", "Audio"], horizontal=True)

if input_mode == "Text":
    # st.form prevents the app from lagging while typing!
    with st.form("text_input_form"):
        user_text = st.text_area("Type your math problem here:")
        # The app will ONLY proceed when this button is clicked
        submitted = st.form_submit_button("Submit Question")
        
        if submitted and user_text:
            st.session_state.raw_text, st.session_state.confidence = process_text(user_text)

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload a photo/screenshot of the problem", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        # Added a button here too for better control
        if st.button("Extract Text from Image"):
            with st.spinner("Extracting text from image..."):
                st.session_state.raw_text, st.session_state.confidence = process_image(uploaded_file)

elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload an audio question", type=["mp3", "wav", "m4a"])
    if audio_file:
        st.audio(audio_file)
        # Added a button here too
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name
                st.session_state.raw_text, st.session_state.confidence = process_audio(tmp_path)
                os.remove(tmp_path)

# --- 2. EXTRACTION PREVIEW & HITL ---
if st.session_state.raw_text:
    st.divider()
    st.subheader("📝 Extraction Preview")
    
    if st.session_state.confidence < 0.8:
        st.warning(f"⚠️ Low confidence extraction ({st.session_state.confidence*100}%). Please verify and edit the text below. (HITL Triggered) [cite: 38, 122]")
    else:
        st.success(f"✅ High confidence extraction ({st.session_state.confidence*100}%)")
        
    edited_text = st.text_area("Edit extracted text if needed:", value=st.session_state.raw_text, height=100)
    
    if st.button("🚀 Solve Problem", type="primary"):
        with st.spinner("Agents are working on the problem..."):
            
            # --- 3. RUN THE MULTI-AGENT PIPELINE ---
            result = run_full_pipeline(edited_text)
            
            st.divider()
            
            if result["status"] == "hitl_required":
                st.error(f"🛑 Human-in-the-Loop (HITL) Triggered: {result['reason']} [cite: 121-125]")
                st.json(result.get("parsed", {}))
                st.info("Please adjust your input text above to be clearer and try again.")
                st.stop()
            
            # --- 4. DISPLAY RESULTS ---
            st.subheader("💡 Solution & Explanation")
            tab1, tab2, tab3, tab4 = st.tabs(["🎓 Explanation", "⚙️ Raw Solution", "📚 RAG Context", "🕵️ Agent Trace"])
            
            with tab1:
                st.markdown(result["final_explanation"])
            with tab2:
                st.text_area("Solver Agent Output:", value=result["raw_solution"], height=200)
            with tab3:
                st.write("Context retrieved from Knowledge Base to help solve this:")
                st.json(result["route"])
            with tab4:
                st.write("**Parsed Problem:**")
                st.json(result["parsed"])
                st.write("**Verification Results:**")
                st.json(result["verification"])
            
            # --- 5. FEEDBACK BUTTONS (MEMORY TRIGGERS) ---
            st.divider()
            st.subheader("Was this helpful?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("☑️ Correct (Save to Memory)"):
                    memory_manager.save_to_memory(
                        original_input=st.session_state.raw_text,
                        parsed=result["parsed"],
                        context=result["route"], 
                        raw_solution=result["raw_solution"],
                        verifier_outcome=result["verification"],
                        feedback="Solution was marked correct by user.",
                        is_correct=True
                    )
                    st.success("Thanks! Saved to successful attempts memory. [cite: 127, 133]")
                    
            with col2:
                if st.button("☒ Incorrect (Trigger Correction)"):
                    st.session_state.show_feedback = True

            if st.session_state.get("show_feedback"):
                feedback_comment = st.text_area("Please provide the correct steps or let us know what went wrong:")
                if st.button("Submit Correction"):
                    memory_manager.save_to_memory(
                        original_input=st.session_state.raw_text,
                        parsed=result["parsed"],
                        context=result["route"], 
                        raw_solution=result["raw_solution"],
                        verifier_outcome=result["verification"],
                        feedback=feedback_comment,
                        is_correct=False
                    )
                    st.success("Correction saved as a learning signal! (HITL Complete) [cite: 127]")
                    st.session_state.show_feedback = False