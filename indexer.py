"""
Questo script esegue la fase di Data Indexing (Indicizzazione dei Dati) per la pipeline RAG. 
Le operazioni svolte sono:
1. Data Loading: Legge i dati grezzi dal file JSON.
2. Chunking a due livelli: Usa MarkdownHeaderTextSplitter per dividere logicamente per sezioni, 
   e RecursiveCharacterTextSplitter per spezzare blocchi troppo lunghi.
3. Debug Export: Salva i chunk generati in un file JSON per ispezione visiva.
4. Embedding Generation & Indexing: Salva i dati in ChromaDB tramite BAAI/bge-m3.
"""

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

        print(f" Errore: Il file {json_path} non esiste")

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