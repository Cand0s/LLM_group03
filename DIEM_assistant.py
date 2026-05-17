import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

# Import the compiled graph from your chatbot.py file
from chatbot import app, AgentState 


# ============================================================
# DIEM Agentic RAG Chatbot — Upgraded UI (Linear Design System)
# ============================================================

def predict(message, history):
    """
    This function is called by Gradio every time the user sends a message.
    """
    # 1. Convert Gradio's history into LangChain's format
    chat_history = []

    if history:
        if isinstance(history[0], dict):
            for item in history:
                if item.get("role") == "user":
                    chat_history.append(HumanMessage(content=item.get("content", "")))
                elif item.get("role") == "assistant":
                    chat_history.append(AIMessage(content=item.get("content", "")))
        else:
            try:
                for human_str, ai_str in history:
                    chat_history.append(HumanMessage(content=human_str))
                    chat_history.append(AIMessage(content=ai_str))
            except (ValueError, TypeError):
                pass
    # 2. Keep history manageable to avoid context overflow
    if len(chat_history) > 15:
        chat_history = chat_history[-15:]

    # 3. Initialize state and execute workflow
    initial_state = AgentState(
        question=message,
        rewritten_question=message,
        chat_history=chat_history,
        documents=[],
        answer="",
        domain_check="",
        retrieval_grade="",
        hallucination_grade="",
        retry_count=0
    )

    try:
        final_state = app.invoke(initial_state)
        return final_state["answer"]
    except Exception as e:
        import traceback
        print(f"Error during graph execution: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

# ─────────────────────────────────────────────
#  CUSTOM CSS (Only for structural fixes)
# ─────────────────────────────────────────────
custom_css = """
/* Prevent global scrolling, hide footer */
footer { display: none !important; }
body, html {margin: 0; padding: 0; height: 100%; }

/* ── CHAT BUBBLES: GENERAL TYPOGRAPHY & SPACING ── */
.message-row .message {
    padding: 12px 18px !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}

/* ── USER BUBBLE (Risolto contrasto testo) ── */
div.message.user, .message-row.user .message, [data-testid="user"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border-radius: 20px 20px 4px 20px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.18) !important;
}

/* Forza il testo bianco anche all'interno dei paragrafi Markdown di Gradio */
div.message.user *, .message-row.user .message *, [data-testid="user"] * {
    color: #FFFFFF !important;
}


/* ── INPUT TEXTBOX (Aumentata la grandezza) ── */
textarea, [data-testid="textbox"] textarea {
    min-height: 65px !important; /* Rende il box più alto */
    padding-top: 18px !important; /* Centra visivamente il testo all'interno */
    padding-bottom: 18px !important;
    font-size: 16px !important;
    border-radius: 24px !important; /* Arrotonda leggermente i bordi per abbinarli alla chat */
}

/* ── EXAMPLES / DATASET CARDS ── */
div[class*="examples"] button, div[data-testid="dataset"] button, .gallery-item {
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-align: center !important;
    white-space: normal !important;
    color: #6366F1 !important; 
    font-weight: 500 !important;
    cursor: pointer !important;
}

.gallery-item:hover, div[class*="examples"] button:hover, div[data-testid="dataset"] button:hover, table.gr-examples tbody tr:hover td {
    background-color: #EEF2FF !important; 
    color: #4338CA !important; 
    border-color: #818CF8 !important; 
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.12) !important;
}
"""

# ─────────────────────────────────────────────
#  NATIVE GRADIO THEME (Corporate Trust)
# ─────────────────────────────────────────────
diem_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E8F0",
    block_radius="16px",
    block_shadow="0 4px 20px -2px rgba(79, 70, 229, 0.10)",
    button_primary_background_fill="#4F46E5",       # Indigo 600
    button_primary_background_fill_hover="#4338CA", # Indigo 700
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F1F5F9",
    input_background_fill="#F8FAFC",
    input_border_color="#E2E8F0",
    input_radius="12px"
)

# ─────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────
with gr.Blocks(theme=diem_theme, css=custom_css, fill_height=True) as demo:

    # ── Header with Inline HTML Styling
    # FIX 2: Reduced top padding to prevent pushing the chat too far down
    gr.HTML("""
    <div style="text-align: center; padding: 15px 20px 5px;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 999px; padding: 6px 16px; font-size: 0.8rem; font-weight: 700; color: #4F46E5; text-transform: uppercase; margin-bottom: 12px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: #10B981; display: inline-block; box-shadow: 0 0 8px rgba(16,185,129,0.5);"></span>
            DIEM · University of Salerno
        </div>
        <h1 style="font-size: clamp(1.6rem, 3.5vw, 2.4rem); font-weight: 800; color: #0F172A; margin: 0 0 8px; line-height: 1.15;">
            Your <span style="background: linear-gradient(135deg, #4F46E5, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Agentic AI</span><br>Department Assistant
        </h1>
        <p style="font-size: 1rem; color: #64748B; max-width: 520px; margin: 0 auto; line-height: 1.5;">
            Ask me anything about degree programs, professors, schedules, labs, regulations, and more — powered by Agentic RAG.
        </p>
    </div>
    """)

    # ── Chat Interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.ChatInterface(
                fn=predict,
                chatbot=gr.Chatbot(
                    avatar_images=["https://cdn-icons-png.flaticon.com/512/1077/1077114.png", "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"],
                    show_label=False,
                ),
                textbox=gr.Textbox(placeholder="Ask about courses, professors, facilities…", show_label=False, lines=1, max_lines=4),
                submit_btn="Send →",
                examples=[
                    "What degree programs does DIEM offer?",
                    "What are Professor Capuano's office hours?",
                    "Which equipment is in the Robotics Lab?",
                ],
                cache_examples=False,
                # FIX 4: Explicitly fill height within the ChatInterface
                fill_height=True
            )

    # ── Disclaimer Accordion
    with gr.Row():
        with gr.Column():
            with gr.Accordion("ℹ️ About this assistant & disclaimer", open=False):
                gr.Markdown(
                    "This assistant uses an **Agentic RAG** architecture to retrieve and reason over "
                    "official DIEM institutional data. While we strive for accuracy, always verify "
                    "critical deadlines or requirements on the official "
                    "[DIEM website](https://www.diem.unisa.it/)."
                )

if __name__ == "__main__":
    demo.launch(share=False)