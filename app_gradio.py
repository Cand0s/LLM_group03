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

# A. Custom CSS for a more professional, institutional look
custom_css = """
body, .gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
h1 {
    color: #004b87 !important; /* University blue */
    text-align: center;
    font-weight: bold;
}
"""

# B. Custom Chatbot Component to add Avatars and UI features
custom_chatbot = gr.Chatbot(
    avatar_images=[
        "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",  # User avatar URL
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"   # Bot avatar URL
    ]
)

# C. Build the ChatInterface with CSS using gr.Blocks
with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=predict,
        chatbot=custom_chatbot,
        title="🤖 DIEM Agentic RAG Chatbot",
        # Using HTML inside the description to center it and make it pop
        description=(
            "<div style='text-align: center; margin-bottom: 20px;'>"
            "Welcome! I am the official AI assistant for the <b>Department of Information Engineering (DIEM)</b> "
            "at the University of Salerno.<br>"
            "<i>Ask me about degree programs, courses, professors, schedules, facilities, and official regulations.</i>"
            "</div>"
        ),
        examples=[
            "What degree programs are offered by DIEM?",
            "What are Professor Capuano's office hours?",
            "Which equipment is available in the Robotics Laboratory?",
            "What are the admission requirements for the Master's Degree in Information Engineering for Digital Medicine?"
        ]
    )

if __name__ == "__main__":
    # Start the local web server
    demo.launch(share=False, css=custom_css)