"""
DIEM Chatbot - Vector Store Indexer (Offline ETL Pipeline)

Questo script funge da "motore di indicizzazione" per il sistema RAG. 
Il suo compito è trasformare i dati testuali grezzi in coordinate matematiche 
ricercabili dall'Intelligenza Artificiale, operando in 4 fasi distinte:

1. CARICAMENTO DATI: Legge il file JSON (Data Dump) generato dallo scraper.
2. CHUNKING: Spezzetta i documenti lunghi in frammenti più piccoli e sovrapposti, 
   ottimizzando i testi per la finestra di contesto (memoria) del LLM.
3. EMBEDDING: Elabora ogni frammento con un modello linguistico (BAAI/bge-m3),
   traducendo il testo in "vettori" (liste di numeri che ne catturano il significato).
4. VECTOR STORE: Archivia i vettori e i testi originali in un database locale 
   (ChromaDB), predisponendo il sistema per il recupero rapido delle informazioni.
"""


# ABBIAMO DIVISIO INDEXER DA EXTRACTOR IN MANIERA TALE DA POTER ESEGUIRE L'INDICIZZAZIONE IN MODO INDIPENDENTE, SENZA DOVER RILANCIARE 
# LO SCRAPER OGNI VOLTA CHE VOGLIAMO AGGIORNARE O RIFARE IL DATABASE VETTORIALE.

import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Parametri configurabili
JSON_PATH = "./diem_knowledge_base.json"
DB_DIRECTORY = "./diem_chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def build_db_from_json(json_path=JSON_PATH):
    print(f"\n=== INIZIO INDICIZZAZIONE DA JSON: {json_path} ===")

    # --- 1. Caricamento Dati ---
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Errore: Il file {json_path} non esiste. Corri prima lo scraper (main.py)!")
        return

    # --- 2. Conversione in Oggetti Document ---
    documents = []
    for entry in data:
        doc = Document(
            page_content=entry["text"],
            metadata={
                "source": entry["source"],
                "type": entry["type"]
            }
        )
        documents.append(doc)
    
    print(f"  → Caricati {len(documents)} documenti dal JSON.")

    # --- 3. Chunking (Divisione del testo in frammenti) ---
    print("\n=== FASE: Chunking ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,    
        chunk_overlap=CHUNK_OVERLAP,   
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  → Creati {len(chunks)} frammenti (chunks) pronti per l'indicizzazione.")

    # --- 4. Embedding e Creazione Vector Store ---
    print("\n=== FASE: Creazione Vector Store (Chroma) ===")
    print("  → Inizializzazione modello Embedding (BAAI/bge-m3 su CUDA)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("  → Generazione vettori e salvataggio su disco (potrebbe volerci qualche minuto)...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY
    )
    
    print(f"\n=== OPERAZIONE CONCLUSA ===")
    print(f"  Database Vettoriale creato con successo in '{DB_DIRECTORY}'.")
    return vector_db

if __name__ == "__main__":
    build_db_from_json()