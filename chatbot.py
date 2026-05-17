import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal


# ──────────────────────────────────────────────
# 0. CONFIGURATION PARAMETERS
# ──────────────────────────────────────────────
# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"

# Reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"

# Vector store
VECTORSTORE_PATH = "./diem_chroma_db"
RETRIEVER_K = 50

# LLM models
LLM_MODEL = "llama3.1"
LLM_TEMPERATURE = 0.1
LLM_JUDGE_TEMPERATURE = 0.0

# Retrieval parameters
TOP_K_RERANKED = 15
CONTEXT_LIMIT = 500
CONTEXT_LIMIT_HALLUCINATION = 3000

# Retry settings
MAX_RETRIES = 1

# Chat history
MAX_CHAT_HISTORY = 20


# ──────────────────────────────────────────────
# 1. MODEL AND DB INITIALIZATION
# ──────────────────────────────────────────────
print("Initializing models...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

# Main LLM (generation)
llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
# LLM for graders (faster; classification only)
llm_judge = ChatOllama(model=LLM_MODEL, temperature=LLM_JUDGE_TEMPERATURE)
print("Models loaded!\n")


# ──────────────────────────────────────────────
# 2. GRAPH STATE
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    question: str                   # original question
    rewritten_question: str         # question rewritten for retrieval
    chat_history: List              # conversation history
    documents: List                 # retrieved documents
    answer: str                     # generated answer
    domain_check: str               # "in_domain" | "out_of_domain"
    retrieval_grade: str            # "relevant" | "not_relevant"
    hallucination_grade: str        # "grounded" | "hallucinated"
    retry_count: int                # used for hallucination regeneration loops


# ──────────────────────────────────────────────
# 3. GUARDRAIL IN – Domain Classifier
# ──────────────────────────────────────────────
domain_check_prompt = ChatPromptTemplate.from_template("""
You are a domain classifier for a university chatbot about DIEM 
(Department of Information Engineering at University of Salerno).

Classify the following question as:
- "in_domain": if it is about DIEM, its courses, professors, facilities, 
  research, degree programs, exams, timetables, regulations, or the 
  University of Salerno.
- "out_of_domain": if it is completely unrelated (e.g., cooking recipes, 
  sports, general knowledge unrelated to DIEM/UNISA).

Reply with ONLY one of these two words, nothing else.

Question: {question}
Classification:""")


def guardrail_input(state: AgentState) -> AgentState:
    """Input guardrail: detect out-of-domain questions."""
    chain = domain_check_prompt | llm_judge | StrOutputParser()
    result = chain.invoke({"question": state["question"]}).strip().lower()
    
    # Normalize the response
    if "out" in result or "out_of_domain" in result:
        domain = "out_of_domain"
    else:
        domain = "in_domain"
    
    print(f"[GUARDRAIL IN] Classification: {domain}")
    return {**state, "domain_check": domain}


def route_after_domain_check(state: AgentState) -> Literal["retrieve", "out_of_domain_response"]:
    if state["domain_check"] == "out_of_domain":
        return "out_of_domain_response"
    return "retrieve"


# ──────────────────────────────────────────────
# 4. RETRIEVAL with re-ranking
# ──────────────────────────────────────────────
def retrieve_and_rerank(state: AgentState) -> AgentState:
    """Retrieve documents and apply re-ranking."""
    query = state.get("rewritten_question", state["question"])
    
    # Initial retrieval
    initial_docs = retriever.invoke(query)
    
    # Re-ranking
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    
    top_k = TOP_K_RERANKED
    top_docs = [doc for doc, score in ranked[:top_k]]
    
    print(f"\n[RETRIEVAL] Top {top_k} documents after re-ranking:")
    for i, (doc, score) in enumerate(ranked[:top_k]):
        print(f"  {i+1}. Score: {score:.2f} | Source: {doc.metadata.get('source', 'N/A')}")
    
    return {**state, "documents": top_docs}


# ──────────────────────────────────────────────
# 5. DOCUMENT GRADER
# ─────────────────────────────────────────────
doc_grade_prompt = ChatPromptTemplate.from_template("""
You are a relevance grader. Evaluate if the retrieved document contains 
information useful to answer the user's question.

Reply ONLY with "relevant" or "not_relevant".

User Question: {question}

Retrieved Document:
{document}

Grade:""")

def grade_documents(state: AgentState) -> AgentState:
    """Evaluate whether retrieved documents are relevant and print their sources."""
    query = state.get("rewritten_question", state["question"])
    docs = state["documents"]

    chain = doc_grade_prompt | llm_judge | StrOutputParser()

    relevant_docs = []
    not_relevant_docs = []

    print(f"\n[DOC GRADER] Evaluating {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'N/A')
        grade = chain.invoke({
            "question": query,
            "document": doc.page_content[:CONTEXT_LIMIT]
        }).strip().lower()

        if "relevant" in grade and "not" not in grade:
            relevant_docs.append(doc)
            status = "✓ RELEVANT"
        else:
            not_relevant_docs.append(doc)
            status = "✗ NOT RELEVANT"

        print(f"  {i:2d}. [{status}] {source}")

    print(f"\n[DOC GRADER SUMMARY] {len(relevant_docs)}/{len(docs)} documents are relevant")

    retrieval_grade = "relevant" if relevant_docs else "not_relevant"
    return {**state, "documents": relevant_docs, "retrieval_grade": retrieval_grade}


# ──────────────────────────────────────────────
# 6. ANSWER GENERATION
# ──────────────────────────────────────────────
system_template = """You are the official AI assistant for DIEM - University of Salerno.

CRITICAL INSTRUCTIONS:
1. Answer using the provided CONTEXT, being as DETAILED and COMPLETE as possible.
2. When describing a laboratory, research group, or course, include ALL available
   details: activities, topics, equipment, people, objectives — nothing omitted.
3. If the question asks about a SPECIFIC person by name, use ONLY information
   about that exact person.
4. If not found in context, say: "I couldn't find information about [topic].
   I recommend visiting www.diem.unisa.it for up-to-date information."
5. Never use phrases like "Based on the context" or "According to...".
6. Always respond in the same language the user used.
"""

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("placeholder", "{chat_history}"),
    ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
])


def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer and track token usage."""
    docs = state.get("documents", [])
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant information found."

    chain = generation_prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"],
        "chat_history": state.get("chat_history", [])
    })

    answer = response.content
    metadata = response.response_metadata
    input_tokens = metadata.get("prompt_eval_count", 0)
    output_tokens = metadata.get("eval_count", 0)
    total_tokens = input_tokens + output_tokens

    print(f"\n[GENERATION] Using {len(docs)} documents for context")
    if docs:
        print(f"[GENERATION SOURCES]:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'N/A')
            print(f"  {i}. {source}")

    print(f"\n[TOKEN USAGE]")
    print(f"  Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}")

    return {**state, "answer": answer}


# ──────────────────────────────────────────────
# 7. GUARDRAIL OUT – hallucination checker
# ──────────────────────────────────────────────
hallucination_prompt = ChatPromptTemplate.from_template("""
You are a fact-checker. Determine if the assistant's answer is grounded 
in the provided context documents, or if it contains information not 
present in the context (hallucination).

Reply ONLY with "grounded" or "hallucinated".

Context Documents:
{context}

Assistant's Answer:
{answer}

Grade:""")

def check_hallucination(state: AgentState) -> AgentState:
    """Output guardrail: check for hallucinations."""
    docs = state.get("documents", [])
    
    if not docs:
        return {**state, "hallucination_grade": "grounded"}
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    chain = hallucination_prompt | llm_judge | StrOutputParser()
    grade = chain.invoke({
        "context": context[:CONTEXT_LIMIT_HALLUCINATION],  # limit for performance
        "answer": state["answer"]
    }).strip().lower()
    
    hallucination_grade = "hallucinated" if "hallucinated" in grade else "grounded"
    print(f"[GUARDRAIL OUT] Hallucination check: {hallucination_grade}")
    
    return {**state, "hallucination_grade": hallucination_grade}


def route_after_hallucination_check(state: AgentState) -> Literal["end", "regenerate"]:
    retry_count = state.get("retry_count", 0)
    if state["hallucination_grade"] == "hallucinated" and retry_count < MAX_RETRIES:
        return "regenerate"
    return "end"


# ──────────────────────────────────────────────
# 8. SPECIAL NODES
# ──────────────────────────────────────────────
def out_of_domain_response(state: AgentState) -> AgentState:
    """Answer for out-of-domain questions."""
    answer = ("I'm the DIEM assistant and I can only answer questions about "
              "the Department of Information Engineering at the University of Salerno. "
              "Your question seems to be outside my area of knowledge. "
              "Please ask me something related to DIEM!")
    return {**state, "answer": answer}


def regenerate_answer(state: AgentState) -> AgentState:
    """Regenerate the answer more conservatively."""
    print("[REGENERATE] Hallucinated answer; regenerating with stricter instructions...")
    docs = state.get("documents", [])
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
    
    strict_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template + "\n\nEXTRA: Be very conservative. "
         "If you are not 100% sure the information is in the context, "
         "say you don't have that specific information."),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
    ])
    
    chain = strict_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"],
        "chat_history": []
    })

    return {**state, "answer": answer, "retry_count": state.get("retry_count", 0) + 1}


# ──────────────────────────────────────────────
# 9. GRAPH CONSTRUCTION
# ──────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("guardrail_input",        guardrail_input)
workflow.add_node("out_of_domain_response", out_of_domain_response)
workflow.add_node("retrieve",               retrieve_and_rerank)
workflow.add_node("grade_documents",        grade_documents)
workflow.add_node("generate",               generate_answer)
workflow.add_node("check_hallucination",    check_hallucination)
workflow.add_node("regenerate",             regenerate_answer)

workflow.set_entry_point("guardrail_input")

workflow.add_conditional_edges(
    "guardrail_input",
    route_after_domain_check,
    {
        "retrieve":               "retrieve",
        "out_of_domain_response": "out_of_domain_response"
    }
)
workflow.add_edge("out_of_domain_response", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", "check_hallucination")

workflow.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination_check,
    {
        "end":        END,
        "regenerate": "regenerate"
    }
)
workflow.add_edge("regenerate", END)

app = workflow.compile()


# ──────────────────────────────────────────────
# 10. MAIN LOOP
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("--- DIEM Agentic RAG Chatbot ---")
    print("Type 'exit' to quit.\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Initial state
        initial_state = AgentState(
            question=user_input,
            rewritten_question=user_input,
            chat_history=chat_history,
            documents=[],
            answer="",
            domain_check="",
            retrieval_grade="",
            hallucination_grade="",
            retry_count=0
        )
        
        # Run the graph
        print("-" * 50)
        final_state = app.invoke(initial_state)
        print("-" * 50)
        
        answer = final_state["answer"]
        print(f"\nBot: {answer}\n")
        
        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        # Keep history manageable
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history = chat_history[-MAX_CHAT_HISTORY:]