"""
indexer_parent_child.py

Data indexing phase with a Parent-Child Chunking strategy for an agentic RAG pipeline.

PARENT-CHILD CHUNKING STRATEGY
"""

import json
import re
import hashlib
import shutil
import torch
import ftfy
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



# ========================
# CONFIGURATION 
# ========================
_FTFY_AVAILABLE = True
JSON_PATH              = "./diem_knowledge_base.json"

# These paths and names must match those in chatbot.py
CHILD_DB_DIRECTORY     = "./diem_chroma_db/children"
PARENT_DB_DIRECTORY    = "./diem_chroma_db/parents"
CHILD_COLLECTION_NAME  = "diem_children"
PARENT_COLLECTION_NAME = "diem_parents"

# i parent non vengono piu' embeddati in Chroma.
# Vengono salvati su disco come JSON e ricaricati in RAM all'avvio del chatbot.
# Stima occupazione: ~10.000 parent x ~4 KB = ~40 MB RAM — trascurabile.
PARENT_STORE_PATH     = "./diem_chroma_db/parent_store.json"

CHUNK_CONFIG = {
    "html": {
        "parent_size": 1250,
        "parent_overlap": 100,
        "child_size": 400,
        "child_overlap": 80
    },
    "pdf": {
        "parent_size": 1700,   # Larger fallback to accommodate long legal sections/tables
        "parent_overlap": 300,
        "child_size": 500,     # Larger children to capture dense academic language
        "child_overlap": 120
    }
}

# Embedding model — identical to the old indexer for compatibility
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug: save a JSON with parent/child samples for visual inspection
DEBUG_EXPORT = True
DEBUG_EXPORT_PATH = "./debug_parent_child_chunks.json"
DEBUG_SAMPLE_SIZE = 50  # quante coppie parent/child esportare

# Flag to enable conditional debug prints
debug = True

# Min length (chars) to retain a parent chunk
PARENT_MIN_CHARS = 80


def _clean_previous_outputs() -> None:
    """Remove persisted indexing outputs so each run starts from a clean slate."""
    child_path = Path(CHILD_DB_DIRECTORY)
    if child_path.exists():
        shutil.rmtree(child_path)
        if debug:
            print(f"    [DEBUG] Removed previous child DB directory: {child_path}")

    parent_store_path = Path(PARENT_STORE_PATH)
    if parent_store_path.exists():
        parent_store_path.unlink()
        if debug:
            print(f"    [DEBUG] Removed previous parent store file: {parent_store_path}")


class ParentStore:
    """
    Dizionario persistente {parent_id -> Document} salvato su disco come JSON.
    Sostituisce il secondo Chroma (parent_db) eliminato con FIX #1.
    I parent non hanno bisogno di embedding: vengono recuperati sempre per
    ID esatto, mai per similarita' semantica.
    Utilizzo nell'indicizzatore:
        store = ParentStore(PARENT_STORE_PATH)
        store.add(parents)
        store.save()

    Utilizzo nel chatbot (all'avvio, una sola volta):
        store = ParentStore.load(PARENT_STORE_PATH)
        doc   = store.get(parent_id)   # O(1)
    """

    def __init__(self, path: str):
        self.path: str = path
        self._store: Dict[str, Document] = {}

    def add(self, parents: List[Document]) -> None:
        """Aggiunge (o sovrascrive) i parent nel dizionario in memoria."""
        for p in parents:
            pid = p.metadata.get("parent_id")
            if pid:
                self._store[pid] = p

    def get(self, parent_id: str) -> Document | None:
        """Restituisce il Document corrispondente a parent_id, o None se assente."""
        return self._store.get(parent_id)

    def save(self) -> None:
        """Serializza il dizionario su disco come JSON."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            pid: {
                "page_content": doc.page_content,
                "metadata":     doc.metadata,
            }
            for pid, doc in self._store.items()
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"    ParentStore saved: {len(self._store)} entries => '{self.path}'")

    @classmethod
    def load(cls, path: str) -> "ParentStore":
        """
        Carica il ParentStore da disco.
        Da chiamare UNA SOLA VOLTA all'avvio del chatbot.
        """
        store = cls(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store._store = {
            pid: Document(page_content=entry["page_content"], metadata=entry["metadata"])
            for pid, entry in data.items()
        }
        print(f"    ParentStore loaded: {len(store._store)} entries from '{path}'")
        return store

    def __len__(self) -> int:
        return len(self._store)
    
    
    
    
# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def _fix_encoding(text: str) -> str:
    """
    Corregge encoding corrotto (¿, Ã¨, L¿obiettivo, ecc.)
    usando ftfy se disponibile, altrimenti un fallback manuale.
    """
    if _FTFY_AVAILABLE:
        return ftfy.fix_text(text)
    # Fallback manuale per i casi più comuni di Latin-1 letto come UTF-8
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


# Mappa manuale source_url -> titolo human-readable.
# Aggiungi qui le URL più frequenti per override esplicito.
_SOURCE_TITLE_OVERRIDES: Dict[str, str] = {
    "https://www.diem.unisa.it/ricerca/progetti-finanziati":  "DIEM - Progetti di Ricerca Finanziati",
    "https://www.diem.unisa.it":                              "DIEM - Sito Ufficiale",
    "https://corsi.unisa.it":                                 "UniSa - Corsi di Studio",
    "https://docenti.unisa.it":                               "UniSa - Docenti",
    "https://www.unisa.it":                                   "UniSa - Sito Ufficiale",
    "https://web.unisa.it":                                   "UniSa - Portale Web",
}


# Segmenti di URL da ignorare nella ricostruzione automatica del titolo
_URL_NOISE_SEGMENTS = re.compile(
    r"^(id|module|row|rescue|page|dettaglio|unisa-rescue-page|\d+)$", re.IGNORECASE
)


def _make_readable_title(source_url: str) -> str:
    """
    Converte un URL grezzo in un titolo leggibile.

    Priorità:
        1. Override manuale esatto (_SOURCE_TITLE_OVERRIDES)
        2. Match parziale prefisso (source inizia con una chiave)
        3. Ricostruzione automatica dai segmenti path
            (kebab-case -> Title Case, segmenti rumorosi filtrati)

    Esempi:
        https://www.diem.unisa.it/ricerca/progetti-finanziati
            -> "DIEM - Progetti di Ricerca Finanziati"  (override)
        https://docenti.unisa.it/070141/home
            -> "UniSa - Docenti > Home"                (match parziale)
        https://corsi.unisa.it/uploads/rescue/__regolamenti-cds/2024/06227.pdf
            -> "UniSa - Corsi di Studio > Regolamenti Cds 2024"
    """
    if not source_url or source_url == "unknown":
        return "Documento Sconosciuto"

    # 1. Override esatto
    if source_url in _SOURCE_TITLE_OVERRIDES:
        return _SOURCE_TITLE_OVERRIDES[source_url]

    # 2. Override parziale (prefisso)
    base_title = None
    for key, val in _SOURCE_TITLE_OVERRIDES.items():
        if source_url.startswith(key + "/") or source_url.startswith(key):
            base_title = val
            break

    # 3. Ricostruzione automatica dal path
    try:
        path = re.sub(r"https?://[^/]+", "", source_url)
        path = path.split("?")[0].split("#")[0].rstrip("/")
        segments = [s for s in path.split("/") if s]
        clean_segments = [
            s for s in segments
            if not _URL_NOISE_SEGMENTS.match(s)
            and not s.startswith("__")
            and not re.match(r"^\d{4,}$", s)
        ]
        readable_parts = []
        for seg in clean_segments:
            seg = re.sub(r"\.(pdf|html|htm|php|aspx)$", "", seg, flags=re.IGNORECASE)
            seg = re.sub(r"[-_]+", " ", seg)
            seg = seg.title()
            readable_parts.append(seg)

        if readable_parts:
            path_label = " > ".join(readable_parts)
            return f"{base_title} > {path_label}" if base_title else path_label
        else:
            return base_title or source_url
    except Exception:
        return base_title or source_url


def _make_section_context(parent_metadata: dict, source_title: str) -> str:
    """
    Costruisce la stringa "\\nSection: ..." per il context injection.

    esclude dal breadcrumb qualsiasi header che:
        - coincida (case-insensitive) con il titolo della source (ridondante)
        - sia una URL grezza
        - sia uguale al breadcrumb dell'header precedente (evita ripetizioni)

    Restituisce stringa vuota se il breadcrumb risultante è vuoto o identico al titolo.
    """
    h_values = [
        parent_metadata.get("Header 1", ""),
        parent_metadata.get("Header 2", ""),
        parent_metadata.get("Header 3", ""),
        parent_metadata.get("Header 4", ""),
        parent_metadata.get("Header 5", ""),
    ]
    _norm_source = source_title.strip().lower()

    def _keep(h: str) -> bool:
        if not h:
            return False
        h_norm = h.strip().lower()
        if h_norm == _norm_source:
            return False
        if h_norm.startswith("http://") or h_norm.startswith("https://"):
            return False
        return True

    parts = [h for h in h_values if _keep(h)]
    if not parts:
        return ""
    hierarchy = " > ".join(parts)
    if hierarchy.strip().lower() == _norm_source:
        return ""
    return f"\nSection: {hierarchy}"




def load_documents(json_path: str) -> List[Document]:
    """
    Carica il JSON della knowledge base e lo converte in LangChain Documents.

    Fix applicati:
        FIX-A  Corregge encoding corrotto con ftfy
        FIX-B  Salva il titolo leggibile in metadata["title"]
        FIX-C  Filtra record con testo troppo corto (< PARENT_MIN_CHARS chars)
    """
    print(f"\n[1/5] Loading data from: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {json_path}")

    if debug:
        print(f"    [DEBUG] Raw JSON entries: {len(data)}")

    documents = []
    skipped_empty = 0
    skipped_short = 0

    for entry_index, entry in enumerate(data):
        raw_text = entry.get("text", "").strip()

        # FIX-C: salta record senza testo
        if not raw_text:
            skipped_empty += 1
            continue

        # FIX-A: correggi encoding corrotto
        clean_text = _fix_encoding(raw_text)

        # FIX-C: salta record troppo corti (es. sola URL iniettata come H1)
        if len(clean_text) < PARENT_MIN_CHARS:
            skipped_short += 1
            if debug:
                print(f"    [DEBUG] Skipped short record ({len(clean_text)} chars): "
                        f"{entry.get('source', '?')[:80]}")
            continue

        source_url = entry.get("source", "unknown")

        # FIX-B: calcola titolo leggibile e salvalo nei metadati
        readable_title = _make_readable_title(source_url)

        doc = Document(
            page_content=clean_text,
            metadata={
                "source":       source_url,
                "title":        readable_title,   # FIX-B
                "type":         entry.get("type", "unknown"),
                "record_index": entry_index,
            }
        )
        documents.append(doc)

    print(f"    Loaded {len(documents)} valid documents "
            f"(skipped: {skipped_empty} empty, {skipped_short} too short).")
    return documents




def split_into_parents(documents: List[Document]) -> List[Document]:
    """
    Split documents into Parent Chunks using Markdown headers first,
    with a character-based fallback for oversized sections.

    Fix applicati:
        FIX-B  Usa metadata["title"] nel Document: header (non l'URL grezzo)
        FIX-C  Filtra parent troppo corti dopo lo split
        FIX-E  Separatori sentence-aware nel fallback splitter
        FIX-F  _make_section_context() deduplica breadcrumb ridondanti
    """
    print(f"\n[2/5] Creating Parent Chunks via Structural Splitter...")

    headers_to_split_on = [
        ("#",     "Header 1"),
        ("##",    "Header 2"),
        ("###",   "Header 3"),
        ("####",  "Header 4"),
        ("#####", "Header 5"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    parents: List[Document] = []
    skipped_short = 0

    for document in documents:
        doc_type       = document.metadata.get("type", "html")
        source_url     = document.metadata.get("source", "unknown")
        # FIX-B: usa il titolo leggibile come H1 root
        readable_title = document.metadata.get("title", _make_readable_title(source_url))

        document.page_content = f"# {readable_title}\n\n{document.page_content.lstrip()}"

        try:
            structural_chunks = markdown_splitter.split_text(document.page_content)
            if not structural_chunks:
                structural_chunks = [document]
        except Exception:
            if debug:
                print(f"    [DEBUG] MarkdownHeaderTextSplitter failed for '{source_url}'. Fallback.")
            structural_chunks = [document]
 
        # FIX-E: separatori sentence-aware nel fallback
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CONFIG[doc_type]["parent_size"],
            chunk_overlap=CHUNK_CONFIG[doc_type]["parent_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,  # FIX-E
        )
 
        for chunk in structural_chunks:
            combined_metadata = {**document.metadata, **chunk.metadata}
            chunk.metadata = combined_metadata
 
            if len(chunk.page_content) > CHUNK_CONFIG[doc_type]["parent_size"]:
                split_fallback = fallback_splitter.split_documents([chunk])
                for fb_chunk in split_fallback:
                    for key in ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5"]:
                        if key in combined_metadata and key not in fb_chunk.metadata:
                            fb_chunk.metadata[key] = combined_metadata[key]
                    # FIX-C: scarta sub-chunk troppo corti
                    if len(fb_chunk.page_content.strip()) >= PARENT_MIN_CHARS:
                        parents.append(fb_chunk)
                    else:
                        skipped_short += 1
            else:
                # FIX-C: scarta chunk troppo corti
                if len(chunk.page_content.strip()) >= PARENT_MIN_CHARS:
                    parents.append(chunk)
                else:
                    skipped_short += 1

    if debug:
        print(f"    [DEBUG] Skipped {skipped_short} parent chunks too short (< {PARENT_MIN_CHARS} chars).")

    # Assign deterministic IDs e inietta Document: header con titolo leggibile
    chunk_counter: Dict[str, int] = defaultdict(int)
    _old_doc_prefix = re.compile(r"^Document:.*?\n\n", re.DOTALL)

    for parent in parents:
        source_url     = parent.metadata.get("source", "unknown")
        readable_title = parent.metadata.get("title", _make_readable_title(source_url))
        # FIX-B: inietta il titolo leggibile, rimuovendo eventuali header precedenti
        _prefix = f"Document: {readable_title}\n\n"
        if not parent.page_content.startswith(_prefix):
            stripped = _old_doc_prefix.sub("", parent.page_content, count=1)
            parent.page_content = f"{_prefix}{stripped}"
            
        chunk_pos = chunk_counter[source_url]
        chunk_counter[source_url] += 1
        hash_string = f"{source_url}::{chunk_pos}::{parent.page_content}"
        parent.metadata["parent_id"] = hashlib.sha256(hash_string.encode("utf-8")).hexdigest()

    print(f"    Created {len(parents)} parent chunks (after enrichment).")
    _print_chunk_stats(parents, "Parent")

    if debug and parents:
        print(f"    [DEBUG] Sample parent_id : {parents[0].metadata['parent_id']}")
        print(f"    [DEBUG] Sample title     : {parents[0].metadata.get('title')}")
        structural_meta = {k: v for k, v in parents[0].metadata.items() if k.startswith("Header")}
        print(f"    [DEBUG] Structural Meta  : {structural_meta}")

    return parents


def split_into_children(parents: List[Document]) -> List[Document]:
    """
    Split each Parent Chunk into smaller Child Chunks.
    """
    children: List[Document] = []
    skipped_short_parent = 0
    skipped_boilerplate  = 0

    # FIX-E: separatori sentence-aware con keep_separator
    child_splitters = {
        doc_type: RecursiveCharacterTextSplitter(
            chunk_size=type_config["child_size"],
            chunk_overlap=type_config["child_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
            keep_separator=True,  # FIX-E
        )
        for doc_type, type_config in CHUNK_CONFIG.items()
    }

    print(f"\n[3/5] Creating Child Chunks...")
    
    for parent in parents:
        doc_type = parent.metadata.get("type", "html")
        config   = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["html"])
        child_splitter = child_splitters[doc_type]
        source_url     = parent.metadata.get("source", "unknown")
        
        # FIX-B: usa il titolo leggibile
        readable_title = parent.metadata.get("title", _make_readable_title(source_url))

        # Rimuovi il prefix "Document: <title>\n\n" per evitare double-header
        _doc_prefix = f"Document: {readable_title}\n\n"
        parent_text = (
            parent.page_content[len(_doc_prefix):]
            if parent.page_content.startswith(_doc_prefix)
            else parent.page_content
        )

        # FIX-F: breadcrumb section deduplicato
        section_context = _make_section_context(parent.metadata, readable_title)

        # Caso: parent già corto -> diventa un unico child
        if len(parent_text) <= config["child_size"]:
            enriched_text = (
                f"Document: {readable_title}{section_context}\nContent: {parent_text}"
            )
            child = Document(page_content=enriched_text, metadata={**parent.metadata})
            child.metadata["child_id"] = hashlib.sha256(
                f"{parent.metadata['parent_id']}::c0::{parent_text}".encode("utf-8")
            ).hexdigest()
            child.metadata["short_doc"] = True
            children.append(child)
            skipped_short_parent += 1
            continue

        # Caso normale: split del parent in più child
        _stripped_parent = Document(page_content=parent_text, metadata=parent.metadata)
        child_docs = child_splitter.split_documents([_stripped_parent])

        for child_index, child in enumerate(child_docs):
            raw_child_text = child.page_content

            enriched_text = (
                f"Document: {readable_title}{section_context}\nContent: {raw_child_text}"
            )
            child.metadata["child_id"] = hashlib.sha256(
                f"{child.metadata['parent_id']}::c{child_index}::{raw_child_text}".encode("utf-8")
            ).hexdigest()
            child.page_content          = enriched_text
            child.metadata["short_doc"] = False
            children.append(child)

    print(f"    Created {len(children)} child chunks.")
    if debug:
        print(f"    [DEBUG] Short parents passed as single child : {skipped_short_parent}")
        print(f"    [DEBUG] Boilerplate table children filtered  : {skipped_boilerplate}")
    _print_chunk_stats(children, "Child")
    if debug and children:
        print(f"    [DEBUG] Sample child_id : {children[0].metadata.get('child_id')}")
        print(f"    [DEBUG]        parent_id: {children[0].metadata.get('parent_id')}")
    return children



def _print_chunk_stats(chunks: List[Document], label: str):
    """Print descriptive statistics about chunk length."""
    lengths = [len(c.page_content) for c in chunks]
    print(f"    [{label}] min={min(lengths)} | " f"max={max(lengths)} | " f"media={int(sum(lengths)/len(lengths))} chars")


def _index_documents_with_progress(db: Chroma, documents: List[Document], ids: List[str], label: str,batch_size: int = 500,):
    """Index documents in batches and, in debug mode, show cumulative progress."""
    total = len(documents)
    if total == 0:
        return

    embedding_fn = db._embedding_function
    for start in range(0, total, batch_size):
        end           = min(start + batch_size, total)
        batch_docs    = documents[start:end]
        batch_ids     = ids[start:end]
        batch_texts   = [doc.page_content for doc in batch_docs]
        # Genera gli embeddings per il batch
        batch_embeddings = embedding_fn.embed_documents(batch_texts)
        # chromadb non accetta None nei metadati: sostituiamo con stringa vuota
        batch_metadatas = [
            {k: (v if v is not None else "") for k, v in doc.metadata.items()}
            for doc in batch_docs
        ]
        # upsert invece di add — idempotente su run multiple
        db._collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
        )
        if debug:
            print(f"    [DEBUG] {label} upserted {end}/{total}")




def export_debug(parents: List[Document], children: List[Document], path: str, n: int = 5):
    """
    Export a sample of parent/child pairs to JSON for visual inspection.
    Useful to verify that the chunks make semantic sense.
    """
    print(f"\n[4a] Debug export → {path}  (sample of {n} parents with their children)")

    # Build the parent_id → parent map.
    parent_map: Dict[str, Document] = {
        p.metadata["parent_id"]: p for p in parents
    }
    # Group children by parent_id.
    child_map: Dict[str, List[Document]] = {}
    for c in children:
        pid = c.metadata.get("parent_id")
        if pid:
            child_map.setdefault(pid, []).append(c)

    sample_pids = list(parent_map.keys())[:n]
    debug_data = []
    for pid in sample_pids:
        p = parent_map[pid]
        clist = child_map.get(pid, [])
        debug_data.append({
            "parent_id":      pid,
            "source":         p.metadata.get("source"),
            "title":           p.metadata.get("title"),
            "parent_length":  len(p.page_content),
            "parent_preview": p.page_content[:300] + ("…" if len(p.page_content) > 300 else ""),
            "num_children":   len(clist),
            "children": [
                {
                    "child_id":     c.metadata.get("child_id"),
                    "length":       len(c.page_content),
                    "preview":      c.page_content[:150] + ("…" if len(c.page_content) > 150 else ""),
                }
                for c in clist
            ]
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    print(f"    ✓ Debug saved.")



def build_vector_stores(parents: List[Document], children: List[Document], embeddings: HuggingFaceEmbeddings,) -> tuple[Chroma, ParentStore]:
    """
    Costruisce il child ChromaDB e il ParentStore su disco.
    I parent vengono salvati in ParentStore (JSON su disco).
    Solo i child vengono embeddati — dimezza il tempo di indicizzazione.
    Restituisce (child_db, parent_store) al posto di (child_db, parent_db).
    """
    # --- Child DB (Chroma, ricerca semantica) ---
    print(f"\n    => Indexing {len(children)} child chunks in '{CHILD_DB_DIRECTORY}'...")
    child_ids = [c.metadata["child_id"] for c in children]
    child_db  = Chroma(
        collection_name=CHILD_COLLECTION_NAME,
        persist_directory=CHILD_DB_DIRECTORY,
        embedding_function=embeddings,
    )
    _index_documents_with_progress(child_db, children, child_ids, "Child chunks")
    print(f"    Child DB ready.")

    # --- Parent Store (JSON, lookup per ID) ---  FIX #1
    print(f"\n    => Saving {len(parents)} parent chunks to ParentStore...")
    parent_store = ParentStore(PARENT_STORE_PATH)
    parent_store.add(parents)
    parent_store.save()
    print(f"    ParentStore ready.")

    return child_db, parent_store



# ============================================================
# ENTRY POINT
# ============================================================

def build_db_from_json(json_path: str = JSON_PATH) -> tuple[Chroma, ParentStore]:
    print("=" * 60)
    print("  RAG INDEXING -- Parent-Child Chunking")
    print("=" * 60)

    _clean_previous_outputs()

    documents = load_documents(json_path)
    parents   = split_into_parents(documents)
    children  = split_into_children(parents)

    if DEBUG_EXPORT:
        export_debug(parents, children, DEBUG_EXPORT_PATH, n=DEBUG_SAMPLE_SIZE)

    print(f"\n[5/5] Initializing embeddings ({EMBEDDING_MODEL} on {EMBEDDING_DEVICE})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    child_db, parent_store = build_vector_stores(parents, children, embeddings)

    print("\n" + "=" * 60)
    print("  OPERATION COMPLETED")
    print(f"  Child DB      => '{CHILD_DB_DIRECTORY}'  [{CHILD_COLLECTION_NAME}]")
    print(f"  Parent Store  => '{PARENT_STORE_PATH}'")
    print(f"  Indexed child chunks : {len(children)}")
    print(f"  Saved parent chunks  : {len(parents)}")
    print("=" * 60)
    
    return child_db, parent_store




if __name__ == "__main__":
    build_db_from_json()