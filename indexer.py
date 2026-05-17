"""
indexer.py — Data Indexing per la pipeline RAG DIEM.

Operazioni:
1. Caricamento dati dal JSON
2. Chunking con RecursiveCharacterTextSplitter
3. Export chunk in JSON per debug
4. Embedding + salvataggio ChromaDB
"""

import json
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

JSON_PATH        = "./diem_knowledge_base.json"
DB_DIRECTORY     = "./diem_chroma_db"
CHUNKS_JSON_PATH = "./diem_chunks_debug.json"

EMBED_MODEL  = "BAAI/bge-m3"
EMBED_DEVICE = "cuda"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

def chunk_documents(documents: list[Document]) -> list[Document]:
    print("\n=== CHUNKING ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  → {len(documents)} documenti → {len(chunks)} chunk")
    return chunks


# ---------------------------------------------------------------------------
# DEBUG EXPORT
# ---------------------------------------------------------------------------

def export_chunks_json(chunks: list[Document]) -> None:
    lengths = [len(c.page_content) for c in chunks]
    output = {
        "statistics": {
            "total_chunks": len(chunks),
            "avg_chars":    round(float(np.mean(lengths)), 0),
            "min_chars":    min(lengths),
            "max_chars":    max(lengths),
            "total_words":  sum(len(c.page_content.split()) for c in chunks),
        },
        "chunks": [
            {
                "id":       i,
                "content":  c.page_content,
                "metadata": c.metadata,
                "chars":    len(c.page_content),
            }
            for i, c in enumerate(chunks)
        ],
    }
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== DEBUG EXPORT ===")
    print(f"  → Salvati {len(chunks)} chunk in '{CHUNKS_JSON_PATH}'")
    print(f"  → Media: {output['statistics']['avg_chars']:.0f} chars/chunk")
    print(f"  → Range: {output['statistics']['min_chars']} – {output['statistics']['max_chars']} chars")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_db(json_path: str = JSON_PATH) -> None:
    print(f"\n=== INDICIZZAZIONE DA: {json_path} ===")

    # 1. Caricamento
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  [ERRORE] File non trovato: {json_path}")
        return

    documents = [
        Document(
            page_content=entry["text"],
            metadata={"source": entry["source"], "type": entry["type"]},
        )
        for entry in data
        if entry.get("text", "").strip()
    ]
    print(f"  → {len(documents)} documenti caricati")

    # 2. Chunking
    chunks = chunk_documents(documents)

    # 3. Export debug
    export_chunks_json(chunks)

    # 4. Embeddings
    print(f"\n=== EMBEDDINGS ({EMBED_MODEL} su {EMBED_DEVICE}) ===")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 5. ChromaDB
    print(f"\n=== SALVATAGGIO CHROMADB in '{DB_DIRECTORY}' ===")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY,
    )
    print(f"\n Completato — {len(chunks)} chunk salvati in '{DB_DIRECTORY}'.")


if __name__ == "__main__":
    build_db()