from langgraph.graph import StateGraph, END

def fetch_web_context(state):
    state["context_web"] = state["tools"].get("web_context") or ""
    return state

def fetch_file_context(state):
    state["context_file"] = state["tools"].get("file_context") or ""
    return state

def generate_answer(state):
    tools = state["tools"]
    prompt = f"Context:\n{tools['context_web']}\n{tools['context_file']}\nQuestion: {state['query']}\nAnswer:"
    resp = tools["chatbot"](prompt)[0]["generated_text"].strip()
    state["answer"] = resp
    return state

def build_agent():
    g = StateGraph()
    g.add_node("web", fetch_web_context)
    g.add_node("file", fetch_file_context)
    g.add_node("answer", generate_answer)
    g.add_edge("web", "file")
    g.add_edge("file", "answer")
    g.add_edge("answer", END)
    return g.compile()
