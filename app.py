import torch
import mlflow
import streamlit as st
from transformers import pipeline
from agent_graph import build_agent
from retrieval import fetch_context
from utils import parse_pdf, parse_html, speak_response
import speech_recognition as sr

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
st.title("🧠 Agentic RAG Voice/Text Assistant")

# ------------------ Load Agent + LLM ------------------
agent = build_agent()

@st.cache_resource
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1" if torch.cuda.is_available() else "tiiuae/falcon-rw-1b"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-generation", model=model_id, tokenizer=model_id, device=device, max_new_tokens=200)

chatbot = load_model()
st.success(f"✅ Model Loaded: {chatbot.model.name_or_path} ({'GPU' if torch.cuda.is_available() else 'CPU'})")

# ------------------ User Input ------------------
mode = st.radio("Select Input Mode", ["Text", "Voice"], horizontal=True)
query = ""

if mode == "Text":
    query = st.text_input("💬 Enter your question:")
else:
    audio_upload = st.file_uploader("🎤 Upload your voice (WAV/MP3):", type=["wav", "mp3"])
    if audio_upload:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_upload) as source:
            audio = recognizer.record(source)
        try:
            query = recognizer.recognize_google(audio)
            st.info(f"You said: {query}")
        except:
            st.warning("Could not understand audio. Please try again.")

# ------------------ Context File Uploads ------------------
pdf_file = st.file_uploader("📄 Upload a PDF document:", type="pdf")
html_file = st.file_uploader("🌐 Upload an HTML file:", type=["html", "htm"])

# ------------------ Response Trigger ------------------
if st.button("🔍 Ask") and query:
    web_context = fetch_context(query)
    file_context = parse_pdf(pdf_file) if pdf_file else parse_html(html_file) if html_file else ""

    mlflow.set_experiment("AgenticRAG")
    with mlflow.start_run():
        mlflow.log_param("query", query)
        mlflow.log_param("web_context", web_context[:250])
        mlflow.log_param("file_context", file_context[:250])

        state = {
            "query": query,
            "tools": {
                "web_context": web_context,
                "file_context": file_context,
                "chatbot": chatbot
            }
        }
        result = agent.invoke(state)
        answer = result["answer"]
        mlflow.log_text(answer, "answer.txt")

    st.markdown(f"**🧠 Answer:** {answer}")
    st.audio(speak_response(answer), format="audio/mp3")
