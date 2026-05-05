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
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# 1. INIZIALIZZAZIONE MODELLI E DB
# ──────────────────────────────────────────────
print("Inizializzazione modelli...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
vectorstore = Chroma(
    persist_directory="./diem_chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# LLM principale (generazione)
llm = ChatOllama(model="llama3.1", temperature=0.1)
# LLM per i grader (più veloce, basta classificare)
llm_judge = ChatOllama(model="llama3.1", temperature=0.0)

print("Modelli caricati!\n")


# ──────────────────────────────────────────────
# 2. STATO DEL GRAFO
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    question: str                   # domanda originale
    rewritten_question: str         # domanda riscritta per il retrieval
    chat_history: List              # storico conversazione
    documents: List                 # documenti recuperati
    answer: str                     # risposta generata
    domain_check: str               # "in_domain" | "out_of_domain"
    retrieval_grade: str            # "relevant" | "not_relevant"
    hallucination_grade: str        # "grounded" | "hallucinated"
    retry_count: int                # numero di tentativi di retrieval


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
    """Guardrail di input: rileva domande fuori dominio."""
    chain = domain_check_prompt | llm_judge | StrOutputParser()
    result = chain.invoke({"question": state["question"]}).strip().lower()
    
    # Normalizza la risposta
    if "out" in result or "out_of_domain" in result:
        domain = "out_of_domain"
    else:
        domain = "in_domain"
    
    print(f"[GUARDRAIL IN] Classificazione: {domain}")
    return {**state, "domain_check": domain}


def route_after_domain_check(state: AgentState) -> Literal["rewrite_query", "out_of_domain_response"]:
    if state["domain_check"] == "out_of_domain":
        return "out_of_domain_response"
    return "rewrite_query"


# ──────────────────────────────────────────────
# 4. QUERY REWRITER – contestualizza domande di follow-up
# ──────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_template("""
You are a STRICTLY SYNTACTIC linguistic analyzer. Your ONLY task is to make the "Follow-up Question" self-contained by resolving pronouns from the chat history.

CRITICAL RULES:
1. NO INVENTIONS: NEVER add adjectives, guess specific names, or insert details the user did not explicitly mention. Do not try to "correct" the user.
2. EXACT MATCH FOR STANDALONE: If the question does not contain pronouns (he, she, it, they, his) or vague references to previous messages, YOU MUST RETURN IT EXACTLY AS IT IS. Do not alter a single word.
3. TOPIC SHIFT: If the user changes the subject entirely, do not drag entities from the previous history into the new question.
4. Return ONLY the final question, nothing else.

Examples:
- History: "Who is Prof. Rossi?" -> Follow-up: "What are his office hours?" -> Rewritten: "What are Prof. Rossi's office hours?"
- History: "Who is Prof. Rossi?" -> Follow-up: "I want to do an Erasmus" -> Rewritten: "I want to do an Erasmus" (CORRECT: Topic shift, history ignored).
- History: "Where is the canteen?" -> Follow-up: "Which equipment is available in the Robotics Laboratory?" -> Rewritten: "Which equipment is available in the Robotics Laboratory?" (CORRECT: No changes made, no adjectives added).

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question:""")




def rewrite_query(state: AgentState) -> AgentState:
    """Riscrive la domanda tenendo conto dello storico."""
    history = state.get("chat_history", [])
    
    # Se non c'è storia, non serve riscrivere
    if not history:
        return {**state, "rewritten_question": state["question"]}
    
    # Formatta la history come stringa
    history_str = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
        for m in history[-6:]  # ultimi 3 scambi
    ])
    
    chain = rewrite_prompt | llm_judge | StrOutputParser()
    rewritten = chain.invoke({
        "chat_history": history_str,
        "question": state["question"]
    }).strip()
    
    print(f"[QUERY REWRITER] Originale: '{state['question']}'")
    print(f"[QUERY REWRITER] Riscritta: '{rewritten}'")
    return {**state, "rewritten_question": rewritten}


# ──────────────────────────────────────────────
# 5. RETRIEVAL con Re-ranking
# ──────────────────────────────────────────────
def retrieve_and_rerank(state: AgentState) -> AgentState:
    """Recupera documenti e applica re-ranking."""
    query = state.get("rewritten_question", state["question"])
    
    # Retrieval iniziale
    initial_docs = retriever.invoke(query)
    
    # Re-ranking
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    
    top_docs = [doc for doc, score in ranked[:5]]
    
    print(f"\n[RETRIEVAL] Top 5 documenti dopo re-ranking:")
    for i, (doc, score) in enumerate(ranked[:5]):
        print(f"  {i+1}. Score: {score:.2f} | Fonte: {doc.metadata.get('source', 'N/A')}")
    
    return {**state, "documents": top_docs}


# ──────────────────────────────────────────────
# 6. DOCUMENT GRADER – valuta rilevanza dei documenti
# ──────────────────────────────────────────────
doc_grade_prompt = ChatPromptTemplate.from_template("""
You are a relevance grader. Evaluate if the retrieved document contains 
information useful to answer the user's question.

Reply ONLY with "relevant" or "not_relevant".

User Question: {question}

Retrieved Document:
{document}

Grade:""")

def grade_documents(state: AgentState) -> AgentState:
    """Valuta se i documenti recuperati sono rilevanti."""
    query = state.get("rewritten_question", state["question"])
    docs = state["documents"]
    
    chain = doc_grade_prompt | llm_judge | StrOutputParser()
    
    relevant_docs = []
    for doc in docs:
        grade = chain.invoke({
            "question": query,
            "document": doc.page_content[:500]  # usa i primi 500 caratteri
        }).strip().lower()
        
        if "relevant" in grade and "not" not in grade:
            relevant_docs.append(doc)
    
    print(f"[DOC GRADER] {len(relevant_docs)}/{len(docs)} documenti rilevanti")
    
    retrieval_grade = "relevant" if relevant_docs else "not_relevant"
    return {**state, "documents": relevant_docs, "retrieval_grade": retrieval_grade}


def route_after_retrieval(state: AgentState) -> Literal["generate", "retry_or_fallback"]:
    retry_count = state.get("retry_count", 0)
    
    if state["retrieval_grade"] == "relevant":
        return "generate"
    elif retry_count < 2:
        return "retry_or_fallback"
    else:
        # Dopo 2 tentativi, genera comunque (con risposta di fallback)
        return "generate"


# ──────────────────────────────────────────────
# 7. RETRY – riformula la query e riprova
# ──────────────────────────────────────────────
retry_prompt = ChatPromptTemplate.from_template("""
The previous search did not return relevant results for this question.
Generate an alternative, more specific search query that might find 
better results in a university department knowledge base.

Original question: {question}
Return ONLY the new search query, nothing else.

Alternative query:""")

def retry_retrieval(state: AgentState) -> AgentState:
    """Genera una query alternativa e riprova il retrieval."""
    retry_count = state.get("retry_count", 0) + 1
    print(f"[RETRY] Tentativo {retry_count}...")
    
    chain = retry_prompt | llm_judge | StrOutputParser()
    new_query = chain.invoke({"question": state["rewritten_question"]}).strip()
    
    print(f"[RETRY] Nuova query: '{new_query}'")
    
    # Retrieval con la nuova query
    initial_docs = retriever.invoke(new_query)
    pairs = [[new_query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:5]]
    
    return {
        **state,
        "rewritten_question": new_query,
        "documents": top_docs,
        "retrieval_grade": "relevant",  # forza il passaggio al grader
        "retry_count": retry_count
    }


# ──────────────────────────────────────────────
# 8. GENERAZIONE risposta
# ──────────────────────────────────────────────
system_template = """You are the official AI assistant for DIEM - University of Salerno.

CRITICAL INSTRUCTIONS:
1. Answer EXCLUSIVELY using the provided CONTEXT.
2. If the question asks about a SPECIFIC person by name, use ONLY 
   information about that exact person. NEVER substitute with information 
   about a different person, even if their data appears in the context.
3. If the specific person is not found in the context, say clearly:
   "Non ho trovato informazioni su [nome] nel database. Ti consiglio di 
   visitare docenti.unisa.it per informazioni aggiornate."
4. Never use phrases like "Based on the context" or "According to...".
5. Always respond in the same language the user used.
"""

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("placeholder", "{chat_history}"),
    ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
])

def generate_answer(state: AgentState) -> AgentState:
    """Genera la risposta finale."""
    docs = state.get("documents", [])
    
    if not docs:
        context = "No relevant information found in the knowledge base."
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
    
    chain = generation_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"],
        "chat_history": state.get("chat_history", [])
    })
    
    return {**state, "answer": answer}


# ──────────────────────────────────────────────
# 9. GUARDRAIL OUT – Hallucination Checker
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
    """Guardrail di output: verifica allucinazioni."""
    docs = state.get("documents", [])
    
    if not docs:
        return {**state, "hallucination_grade": "grounded"}
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    chain = hallucination_prompt | llm_judge | StrOutputParser()
    grade = chain.invoke({
        "context": context[:3000],  # limita per performance
        "answer": state["answer"]
    }).strip().lower()
    
    hallucination_grade = "hallucinated" if "hallucinated" in grade else "grounded"
    print(f"[GUARDRAIL OUT] Hallucination check: {hallucination_grade}")
    
    return {**state, "hallucination_grade": hallucination_grade}


def route_after_hallucination_check(state: AgentState) -> Literal["end", "regenerate"]:
    retry_count = state.get("retry_count", 0)
    if state["hallucination_grade"] == "hallucinated" and retry_count < 1:
        return "regenerate"
    return "end"


# ──────────────────────────────────────────────
# 10. NODI SPECIALI
# ──────────────────────────────────────────────
def out_of_domain_response(state: AgentState) -> AgentState:
    """Risposta per domande fuori dominio."""
    answer = ("I'm the DIEM assistant and I can only answer questions about "
              "the Department of Information Engineering at the University of Salerno. "
              "Your question seems to be outside my area of knowledge. "
              "Please ask me something related to DIEM!")
    return {**state, "answer": answer}


def regenerate_answer(state: AgentState) -> AgentState:
    """Rigenera la risposta in modo più cauto."""
    print("[REGENERATE] Risposta allucinata, rigenero con istruzioni più restrittive...")
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
# 11. COSTRUZIONE DEL GRAFO (LangGraph)
# ──────────────────────────────────────────────
workflow = StateGraph(AgentState)

# Aggiunta nodi
workflow.add_node("guardrail_input",        guardrail_input)
workflow.add_node("out_of_domain_response", out_of_domain_response)
workflow.add_node("rewrite_query",          rewrite_query)
workflow.add_node("retrieve",               retrieve_and_rerank)
workflow.add_node("grade_documents",        grade_documents)
workflow.add_node("retry_retrieval",        retry_retrieval)
workflow.add_node("generate",               generate_answer)
workflow.add_node("check_hallucination",    check_hallucination)
workflow.add_node("regenerate",             regenerate_answer)

# Entry point
workflow.set_entry_point("guardrail_input")

# Edges condizionali
workflow.add_conditional_edges(
    "guardrail_input",
    route_after_domain_check,
    {
        "rewrite_query":           "rewrite_query",
        "out_of_domain_response":  "out_of_domain_response"
    }
)
workflow.add_edge("out_of_domain_response", END)
workflow.add_edge("rewrite_query",          "retrieve")
workflow.add_edge("retrieve",               "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_after_retrieval,
    {
        "generate":           "generate",
        "retry_or_fallback":  "retry_retrieval"
    }
)
workflow.add_edge("retry_retrieval",    "grade_documents")
workflow.add_edge("generate",           "check_hallucination")

workflow.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination_check,
    {
        "end":        END,
        "regenerate": "regenerate"
    }
)
workflow.add_edge("regenerate", END)

# Compila il grafo
app = workflow.compile()


# ──────────────────────────────────────────────
# 12. LOOP PRINCIPALE
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
        
        # Stato iniziale
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
        
        # Esegui il grafo
        print("-" * 50)
        final_state = app.invoke(initial_state)
        print("-" * 50)
        
        answer = final_state["answer"]
        print(f"\nBot: {answer}\n")
        
        # Aggiorna la chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        
        # Mantieni la history gestibile (ultimi 10 scambi)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]