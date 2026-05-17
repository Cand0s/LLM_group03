# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"         
import os
import torch
import json
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

K_CHILDREN = 20  # Number of child documents to retrieve before re-ranking
TOP_K = 10  # Number of top documents to use for generation (after re-ranking)
MAX_CONTEXT_CHARS = 3000        # Max characters from retrieved documents to include in the generation prompt 
MAX_HISTORY_TURNS = 3           # Number of previous conversation turns to include in the query REWRITING prompt (couples of user+bot = 1 turn)
MAX_HISTORY_CHARS = 2000        # Max characters from chat history to include in the query REWRITING prompt (to prevent context overflow)
DOC_GRADING_CHARS = 1000        # Max characters from each document to include in the grading prompt
debug = True

# ──────────────────────────────────────────────
# GLOBAL SWITCHES
# ──────────────────────────────────────────────
# Set these to True or False to enable/disable specific nodes
ENABLE_DOMAIN_CHECK = True
ENABLE_QUERY_REWRITING = True
ENABLE_DOC_GRADING = False
ENABLE_HALLUCINATION_CHECK = False
# Set to True to enable Maximal Marginal Relevance (MMR) for diversity in retrieval; False uses standard similarity search.
ENABLE_MMR = False      


# --- ADAPTIVE RETRIEVAL THRESHOLDS ---
# Define multiplier and absolute fallback
RELATIVE_MULTIPLIER = 0.3
MIN_ABSOLUTE_THRESHOLD = 0.009 

# ──────────────────────────────────────────────
# 1. MODEL AND DB INITIALIZATION
# ──────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)

# Connect to ChromaDB. 
child_db = Chroma(
    persist_directory="./diem_chroma_db/children",
    embedding_function=embeddings,
    collection_name="diem_children",
)

# Load parent store from JSON file into a Python dictionary (O(1) lookups).
try:
    with open("./diem_chroma_db/parent_store.json", "r", encoding="utf-8") as f:
        parent_store = json.load(f)
    assert isinstance(parent_store, dict), "parent_store deve essere un dizionario"
    print(f"Parent store loaded: {len(parent_store)} entries")
except (FileNotFoundError, json.JSONDecodeError, AssertionError) as e:
    raise RuntimeError(f"[FATAL] Impossibile caricare parent_store.json: {e}")

# Main LLM (generation)
llm = ChatOllama(model="llama3.1", temperature=0.1)                 # llm = ChatOllama(model="qwen3:8b", temperature=0.1)   # llm = ChatOllama(model="llama3.1", temperature=0.1)

# LLM for graders (classification only)                             ###TODO consider using a smaller/faster model for grading tasks, llm_judge = ChatOllama(model="llama3.2:3b", temperature=0.0)  
llm_judge = ChatOllama(model="llama3.1", temperature=0.0)
print("Models loaded!\n")





def retrieve_with_parent_context(query: str,child_db: Chroma, parent_store: dict, k_children: int = 20) -> list[Document]:
    """
    Two-stage retrieval:
      1. Ricerca semantica sui child chunk (precisa, vettoriale) - child_db
      2. Fetch dei parent corrispondenti per ID esatto           - parent_store
 
    FIX #1a: parent_store e' un dict Python, non un Chroma.
             Lookup O(1), zero GPU, zero embedding.
 
    FIX #1b: il codice originale usava similarity_search con
             filter={"chunk_id": parent_id}. Due problemi simultanei:
             (a) il campo si chiama "parent_id" non "chunk_id" — non trovava mai nulla;
             (b) similarity_search e' sbagliato per un lookup per ID noto.
             Entrambi causavano fallback silenzioso su ogni query: il LLM
             riceveva sempre i child invece dei parent piu' ricchi.
 
    Args:
        query:        domanda dell'utente
        child_db:     ChromaDB con i child chunk embeddati
        parent_store: dict {parent_id: {"page_content": ..., "metadata": ...}}
                      caricato all'avvio con json.load(open(PARENT_STORE_PATH))
        k_children:   numero di child chunk da recuperare nella fase 1
 
    Returns:
        Lista di Document (parent) da passare al LLM
    """
    # Step 1: Semantic search on children (MMR vs Standard Similarity)
    if ENABLE_MMR:
        # fetch_k determines how many documents to pull initially before applying the MMR algorithm.
        # lambda_mult (0 to 1) balances relevance (1) and diversity (0). 0.5 is a standard balance.
        child_results = child_db.max_marginal_relevance_search(query, k=k_children, fetch_k=k_children * 3, lambda_mult=0.5)
        if debug:
            print(f"\n[retrieval] MMR Search enabled. Query: '{query}'")
    else:
        # Standard similarity search
        child_results = child_db.similarity_search(query, k=k_children)
        if debug:
            print(f"\n[retrieval] Standard Similarity Search. Query: '{query}'")

    if debug:
        print(f"[retrieval] {len(child_results)} child chunks found")

    # Step 2: fetch dei parent per ID — FIX #1a e #1b
    seen_parent_ids: set[str] = set()
    parent_docs: list[Document] = []

    for child in child_results:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        # O(1) Lookup in RAM dictionary — no embeddings, no ranking
        entry = parent_store.get(parent_id)

        if entry is not None:
            parent_docs.append(Document(
                page_content=entry["page_content"],
                metadata=entry["metadata"],
            ))
        else:
            # This should never happen if indexer and chatbot are aligned correctly. Log a warning if it does.
            if debug:
                print(f"[retrieval] WARNING: parent_id {parent_id[:16]}... "f"non trovato nel parent_store — uso child come fallback")
            parent_docs.append(child)
    if debug:
        print(f"[retrieval] {len(parent_docs)} parent chunks returned to the LLM")
        
    return parent_docs

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
    retrieval_retry_count: int      # tracks retrieval retries
    hallucination_retry_count: int  # tracks hallucination regeneration loop


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
    """
        Input guardrail: detect out-of-domain questions.
        If the question is out of domain, route to a special response node.
    """
    chain = domain_check_prompt | llm_judge | StrOutputParser()
    result = chain.invoke({"question": state["question"]}).strip().lower()
    
    # Normalize the response.
    if "out" in result or "out_of_domain" in result:
        domain = "out_of_domain"
    else:
        domain = "in_domain"
    
    print(f"[GUARDRAIL IN] Classification: {domain}")
    return {**state, "domain_check": domain}




def route_after_domain_check(state: AgentState) -> Literal["rewrite_query", "out_of_domain_response"]:
    """
    Route based on domain check result.
    If in-domain, go to query rewriter. If out-of-domain, go to special response.
    """
    if state["domain_check"] == "out_of_domain":
        return "out_of_domain_response"
    return "rewrite_query"


# ──────────────────────────────────────────────
# 4. QUERY REWRITER
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
    """Rewrite the question using the chat history."""

    history = state.get("chat_history", [])
    
    # If there is no history, there is nothing to rewrite.
    if not history:
        return {**state, "rewritten_question": state["question"]}
    
    # Format the history as a string.
    history_str = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
        for m in history[-(MAX_HISTORY_TURNS * 2):]  # last N turns (Human + AI pairs)
    ])
    
    chain = rewrite_prompt | llm_judge | StrOutputParser()
    rewritten = chain.invoke({
        "chat_history": history_str,
        "question": state["question"]
    }).strip()
    
    print(f"[QUERY REWRITER] Original: '{state['question']}'")
    print(f"[QUERY REWRITER] Rewritten: '{rewritten}'")
    return {**state, "rewritten_question": rewritten}

# ──────────────────────────────────────────────
# 5. RETRIEVAL with re-ranking
# ──────────────────────────────────────────────
def retrieve_and_rerank(state: AgentState) -> AgentState:
    """Retrieve with parent-child context and re-ranking."""
    query = state.get("rewritten_question", state["question"])
    
    # Step 1: two-stage retrieval to get parent documents with richer context: retrieve K_CHILDREN children, then expand to parents
    initial_docs = retrieve_with_parent_context(query=query, child_db=child_db, parent_store=parent_store, k_children=K_CHILDREN)

    if not initial_docs:
        return {**state, "documents": []}
    
    # Step 2: re-rank the parents (richer text = better quality)
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

   # Step 3: Relative adaptive threshold logic
    # Since the list is sorted descending, the first element has the highest score
    max_score = ranked[0][1] 
    
    
    
    # Calculate the dynamic threshold (30% of max score)
    calculated_threshold = max_score * RELATIVE_MULTIPLIER
    
    # Ensure the threshold never drops below a safe minimum to avoid garbage retrieval
    current_threshold = max(calculated_threshold, MIN_ABSOLUTE_THRESHOLD)
    
    if debug:
        print(f"\n[RETRIEVAL] Max score found: {max_score:.2f}")
        print(f"[RETRIEVAL] Applying RELATIVE threshold: {current_threshold:.2f} (30% of max, min {MIN_ABSOLUTE_THRESHOLD})")

    # Step 4: Apply the chosen threshold
    top_docs = [doc for doc, score in ranked[:TOP_K] if score >= current_threshold]

    if debug:
        print(f"\n[RETRIEVAL] Kept {len(top_docs)} documents out of top {TOP_K}:")
    
    # Debug print to show kept vs discarded chunks
    if debug: 
        for i, (doc, score) in enumerate(ranked[:TOP_K]):
            content_preview = doc.page_content.replace("\n", " ")[:1000]  # preview first 1000 chars without newlines
            if score >= current_threshold:
                print(f"  {i+1}. [KEPT] Score: {score:.2f} | Source: {doc.metadata.get('source', 'N/A')}")
                print(f"     Preview: {content_preview}...")
            else:
                print(f"  {i+1}. [DISCARDED] Score: {score:.2f} | Source: {doc.metadata.get('source', 'N/A')}")

    return {**state, "documents": top_docs}
# ──────────────────────────────────────────────
# 6. DOCUMENT GRADER – grade document relevance
# ──────────────────────────────────────────────
# Single-call batch prompt: one LLM inference instead of N sequential ones.
doc_grade_prompt = ChatPromptTemplate.from_template("""
You are a relevance grader. For each numbered document below, decide if it \
contains information useful to answer the user's question.

Reply with one line per document in the exact format:
  <number>: relevant
  <number>: not_relevant

Output ONLY these lines, nothing else.

User Question: {question}

{documents}

Grades:""")


def grade_documents(state: AgentState) -> AgentState:
    """Evaluate whether retrieved documents are relevant — single batched LLM call."""
    query = state.get("rewritten_question", state["question"])
    docs = state["documents"]

    if not docs:
        return {**state, "retrieval_grade": "not_relevant"}

    # Build the numbered document list sent to the LLM in one shot
    docs_text = "\n\n".join([
        f"Document {i + 1}:\n{doc.page_content[:DOC_GRADING_CHARS]}"
        for i, doc in enumerate(docs)
    ])

    chain = doc_grade_prompt | llm_judge | StrOutputParser()
    try:
        raw = chain.invoke({"question": query, "documents": docs_text}).strip()

        # Parse "1: relevant\n2: not_relevant\n..." — tolerant to extra whitespace
        relevant_docs = []
        graded_indices = set()
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                idx_str, label = line.split(":", 1)
                idx = int(idx_str.strip()) - 1   # convert to 0-based
                label = label.strip().lower()
                if 0 <= idx < len(docs):
                    graded_indices.add(idx)
                    if "not" not in label and "relevant" in label:
                        relevant_docs.append(docs[idx])
            except (ValueError, IndexError):
                continue  # ignore malformed lines

        # Safety: if the model failed to grade some docs, keep them (same as before)
        for i, doc in enumerate(docs):
            if i not in graded_indices:
                print(f"[DOC GRADER] WARNING: doc {i + 1} not graded — keeping as relevant")
                relevant_docs.append(doc)

        print(f"[DOC GRADER] {len(relevant_docs)}/{len(docs)} relevant documents")
        retrieval_grade = "relevant" if relevant_docs else "not_relevant"
        return {**state, "documents": relevant_docs, "retrieval_grade": retrieval_grade}

    except Exception as e:
        print(f"[ERROR] Document grading failed: {e}")
        return {**state, "documents": docs, "retrieval_grade": "relevant"}


def route_after_retrieval(state: AgentState) -> Literal["generate", "retry_or_fallback"]:
    retrieval_retry_count = state.get("retrieval_retry_count", 0)

    if state["retrieval_grade"] == "relevant":
        print("[ROUTING] Documents relevant → generate")
        return "generate"

    if retrieval_retry_count < 2:
        print(f"[ROUTING] Retry {retrieval_retry_count + 1}/2 → retry_retrieval")
        return "retry_or_fallback"

    print("[ROUTING] Max retries reached → generate (fallback)")
    return "generate"

# ──────────────────────────────────────────────
# 7. RETRY 
# ──────────────────────────────────────────────
retry_prompt = ChatPromptTemplate.from_template("""
The previous search did not return relevant results for this question.
Generate an alternative, more specific search query that might find 
better results in a university department knowledge base.

Original question: {question}
Return ONLY the new search query, nothing else.

Alternative query:""")

def retry_retrieval(state: AgentState) -> AgentState:
    """Generate an alternative query and retry retrieval."""
    retrieval_retry_count = state.get("retrieval_retry_count", 0) + 1
    print(f"[RETRY] Attempt {retrieval_retry_count}...")
    
    chain = retry_prompt | llm_judge | StrOutputParser()
    new_query = chain.invoke({"question": state["question"]}).strip()
    
    print(f"[RETRY] New query: '{new_query}'")
    
    initial_docs = retrieve_with_parent_context(query=new_query, child_db=child_db, parent_store=parent_store, k_children=K_CHILDREN)
    
    if not initial_docs:
        top_docs = []
    else:
        pairs = [[new_query, doc.page_content] for doc in initial_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked[:TOP_K]]
    
    return {
        **state,
        "rewritten_question": new_query,
        "documents": top_docs,
        "retrieval_retry_count": retrieval_retry_count
    }   



# ──────────────────────────────────────────────
# 8. ANSWER GENERATION
# ──────────────────────────────────────────────
system_template = """You are the official AI assistant for DIEM - University of Salerno.

CRITICAL INSTRUCTIONS:
1. Answer EXCLUSIVELY using the provided CONTEXT.
2. If the question asks about a SPECIFIC person by name, use ONLY 
   information about that exact person. NEVER substitute with information 
   about a different person, even if their data appears in the context.
3. If the specific person is not found in the context, say clearly (in the user's language):
    "I couldn't find information about [name] in the database. I recommend visiting docenti.unisa.it for up-to-date information."
4. Never use phrases like "Based on the context" or "According to...".
5. Always respond in the same language the user used.

"""

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("placeholder", "{chat_history}"),
    ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
])

def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer."""
    docs = state.get("documents", [])
    
    if not docs:
        context = "No relevant information found in the knowledge base."
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
        context = context[:MAX_CONTEXT_CHARS]                       # limit context size 
    
    chain = generation_prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({
            "context": context,
            "question": state.get("rewritten_question", state["question"]),         # "question": state["question"],
            "chat_history": state.get("chat_history", [])
        })
        return {**state, "answer": answer}
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        fallback_answer = "I encountered an error while generating the answer. Please try again."
        return {**state, "answer": fallback_answer}


# ──────────────────────────────────────────────
# 9. GUARDRAIL OUT – hallucination checker
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
    try:
        print(f"\n[GUARDRAIL OUT] Checking for hallucinations...")
        grade = chain.invoke({
            "context": context[:MAX_CONTEXT_CHARS],  # limit for performance
            "answer": state["answer"]
        }).strip().lower()
        
        hallucination_grade = "hallucinated" if "hallucinated" in grade else "grounded"
        print(f"[GUARDRAIL OUT] Hallucination check: {hallucination_grade}")
        
        return {**state, "hallucination_grade": hallucination_grade}
    except Exception as e:
        print(f"[ERROR] Hallucination check failed: {e}. Assuming grounded.")
        return {**state, "hallucination_grade": "grounded"}


def route_after_hallucination_check(state: AgentState) -> Literal["end", "regenerate"]:
    """Route after hallucination check: if grounded, end; if hallucinated, decide whether to regenerate based on retry count."""
    hallucination_retry_count = state.get("hallucination_retry_count", 0)
    if state["hallucination_grade"] == "hallucinated" and hallucination_retry_count < 1:
        return "regenerate"
    return "end"


# ──────────────────────────────────────────────
# 10. SPECIAL NODES
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
    
    current_retry_count = state.get("hallucination_retry_count", 0)
    
    try:
        answer = chain.invoke({
            "context": context,
            "question": state.get("rewritten_question", state["question"]),
            "chat_history": state.get("chat_history", [])
        })
        return {
            **state, 
            "answer": answer, 
            "hallucination_retry_count": current_retry_count + 1
        }
    except Exception as e:
        print(f"[ERROR] Answer regeneration failed: {e}")
        return {
            **state, 
            "answer": "Error regenerating answer.", 
            "hallucination_retry_count": current_retry_count + 1
        }


# ──────────────────────────────────────────────
# 11. GRAPH CONSTRUCTION (LangGraph)
# ──────────────────────────────────────────────
workflow = StateGraph(AgentState)

# 1. Core nodes (always active)
workflow.add_node("retrieve", retrieve_and_rerank)
workflow.add_node("generate", generate_answer)

# 2. Domain Check Node & Edges
if ENABLE_DOMAIN_CHECK:
    workflow.add_node("guardrail_input", guardrail_input)
    workflow.add_node("out_of_domain_response", out_of_domain_response)
    
    workflow.set_entry_point("guardrail_input")
    
    # Dynamically determine the next node based on the Query Rewriting flag
    next_node_after_domain = "rewrite_query"
    
    workflow.add_conditional_edges(
        "guardrail_input",
        route_after_domain_check,
        {
            # The routing function returns "rewrite_query". 
            # Here we map it to either the actual rewriting node, or skip directly to retrieval.
            "rewrite_query": next_node_after_domain, 
            "out_of_domain_response": "out_of_domain_response"
        }
    )
    workflow.add_edge("out_of_domain_response", END)
else:
    # If Domain Check is disabled, set the entry point to the next available active node
    workflow.set_entry_point("rewrite_query")
    

# 3. Query Rewriting Node & Edges
if ENABLE_QUERY_REWRITING:
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_edge("rewrite_query", "retrieve")


# 4. Document Grading Node & Edges
if ENABLE_DOC_GRADING:
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("retry_retrieval", retry_retrieval)
    
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_retrieval,
        {
            "generate": "generate",
            "retry_or_fallback": "retry_retrieval"
        }
    )
    workflow.add_edge("retry_retrieval", "grade_documents")
else:
    # If grading is disabled, go directly from retrieve to generate
    workflow.add_edge("retrieve", "generate")


# 5. Hallucination Check Node & Edges
if ENABLE_HALLUCINATION_CHECK:
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("regenerate", regenerate_answer)
    
    workflow.add_edge("generate", "check_hallucination")
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "end": END,
            "regenerate": "regenerate"
        }
    )
    workflow.add_edge("regenerate", "check_hallucination")
else:
    # If hallucination check is disabled, the generation is the final step
    workflow.add_edge("generate", END)

# Compile the graph.
app = workflow.compile()


# ──────────────────────────────────────────────
# 12. MAIN LOOP
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
        
        # Initial state.
        initial_state = AgentState(
            question=user_input,
            rewritten_question=user_input,
            chat_history=chat_history,
            documents=[],
            answer="",
            domain_check="",
            retrieval_grade="",
            hallucination_grade="",
            retrieval_retry_count=0,       # Initialize retrieval counter to 0
            hallucination_retry_count=0    # Initialize hallucination counter to 0
        )
        
        # Run the graph.
        print("-" * 50)
        final_state = app.invoke(initial_state)
        print("-" * 50)
        
        answer = final_state["answer"]
        print(f"\nBot: {answer}\n")
        
        # Update chat history.
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        
        # Keep the history manageable (last 8 turns).
        # Keep the history manageable (last 8 turns).
        if len(chat_history) > 16:    
            chat_history = chat_history[-16:]
            
        history_text = "".join(m.content for m in chat_history)
        while len(history_text) > MAX_HISTORY_CHARS and len(chat_history) > 2:
            chat_history = chat_history[2:]  # remove the oldest user+bot pair
            history_text = "".join(m.content for m in chat_history)