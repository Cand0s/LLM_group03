import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

# Import the compiled graph from your chatbot.py file
from chatbot import app, AgentState 

def predict(message, history):
    """
    This function is called by Gradio every time the user sends a message.
    """
    # 1. Convert Gradio's history into LangChain's format
    chat_history = []
    
    # Handle different history formats from Gradio 6.0+
    if history:
        if isinstance(history[0], dict):
            # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            for item in history:
                if item.get("role") == "user":
                    chat_history.append(HumanMessage(content=item.get("content", "")))
                elif item.get("role") == "assistant":
                    chat_history.append(AIMessage(content=item.get("content", "")))
        else:
            # Legacy format: [[user_msg, bot_msg], ...]
            try:
                for human_str, ai_str in history:
                    chat_history.append(HumanMessage(content=human_str))
                    chat_history.append(AIMessage(content=ai_str))
            except (ValueError, TypeError):
                # If unpacking fails, try treating as list of single items
                pass
    
    # Keep only the last 10 exchanges to avoid exceeding context limits
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    # 2. Initialize the state for LangGraph
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
    
    # 3. Execute the workflow
    try:
        final_state = app.invoke(initial_state)
        return final_state["answer"]
    except Exception as e:
        print(f"Error during graph execution: {e}")
        return "Sorry, an internal error occurred while processing your request."

# --- UI ENHANCEMENTS ---
# Custom CSS for specific overrides not handled by the Theme
custom_css = """
.header-container {
    text-align: center;
    margin-bottom: 20px;
    padding: 10px;
}
.header-title {
    color: #004b87 !important; /* University blue */
    font-weight: 800;
    margin-bottom: 5px;
}
.header-subtitle {
    color: #555555;
    font-size: 1.1em;
}
"""

# Custom Chatbot Component with improved UX styling
custom_chatbot = gr.Chatbot(
    avatar_images=[
        "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",  # User avatar
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"   # Bot avatar
    ],
    height=550,              # Fixes the height to prevent infinite scrolling
)

# Use a built-in Gradio theme, customizing the primary hue to match the University colors
diem_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="*background_fill_primary",
    block_radius="*radius_lg" # Smoother rounded corners
)

# Build the ChatInterface wrapping it in custom layout rows
with gr.Blocks(theme=diem_theme, css=custom_css) as demo:
    
    # Custom Header Layout
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                "<div class='header-container'>"
                "<h1 class='header-title'>🤖 DIEM Agentic RAG Chatbot</h1>"
                "<div class='header-subtitle'>Welcome! I am the official AI assistant for the <b>Department of Information Engineering (DIEM)</b> at the University of Salerno.</div>"
                "<i>Ask me about degree programs, courses, professors, schedules, facilities, and official regulations.</i>"
                "</div>"
            )

    # Main Chat Interface without the default title/description to keep it clean
    with gr.Row():
        with gr.Column():
            gr.ChatInterface(
                fn=predict,
                chatbot=custom_chatbot,
                examples=[
                    "What degree programs are offered by DIEM?",
                    "What are Professor Capuano's office hours?",
                    "Which equipment is available in the Robotics Laboratory?",
                    "What are the admission requirements for the Master's Degree in Information Engineering for Digital Medicine?"
                ]
            )
            
    # Footer / Disclaimer placed neatly in an expandable accordion
    with gr.Row():
        with gr.Accordion("ℹ️ Additional Information & Disclaimer", open=False):
            gr.Markdown(
                "This chatbot relies on an Agentic RAG architecture to fetch institutional information. "
                "While we strive for accuracy, always verify critical deadlines or requirements on the "
                "official [DIEM University Website](https://corsi.unisa.it/diem)."
            )

if __name__ == "__main__":
    # Start the local web server
    demo.launch(share=False)