import torch, mlflow, streamlit as st
from transformers import pipeline
from io import BytesIO
from agent_graph import build_agent
from retrieval import fetch_context
from utils import parse_pdf, parse_html, speak_response

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
agent = build_agent()

@st.cache_resource
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1" if torch.cuda.is_available() else "tiiuae/falcon-rw-1b"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-generation", model=model_id, tokenizer=model_id, device=device, max_new_tokens=200)

chat = load_model()
st.success(f"Model: {chat.model.name_or_path} ({'GPU' if torch.cuda.is_available() else 'CPU'})")

mode = st.radio("Input Mode", ["Text", "Voice"], horizontal=True)
query = ""
if mode == "Text":
    query = st.text_input("Your question:")
else:
    upl = st.file_uploader("Upload voice (wav/mp3):", type=["wav", "mp3"])
    if upl:
        import speech_recognition as sr
        rec = sr.Recognizer()
        with sr.AudioFile(upl) as src: audio = rec.record(src)
        query = rec.recognize_google(audio) if audio else ""

pdf = st.file_uploader("Upload PDF:", type="pdf")
html = st.file_uploader("Upload HTML:", type=["html","htm"])

if st.button("Ask") and query:
    webctx = fetch_context(query)
    filectx = parse_pdf(pdf) if pdf else parse_html(html) if html else ""
    mlflow.set_experiment("AgenticRAG")
    with mlflow.start_run():
        mlflow.log_param("query", query)
        mlflow.log_param("web_context", webctx[:250])
        mlflow.log_param("file_context", filectx[:250])

        state = {"query": query, "tools": {"web_context": webctx, "file_context": filectx, "chatbot": chat}}
        result = agent.invoke(state)
        answer = result["answer"]
        mlflow.log_text(answer, "answer.txt")

    st.markdown(f"**🧠 Answer:** {answer}")
    st.audio(speak_response(answer), format="audio/mp3")
