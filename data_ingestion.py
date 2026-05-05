"""
DIEM Chatbot - Web Scraper & Vector Store Builder (Advanced Architecture)
"""

import time
import io
import os
import requests
import urllib3
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Risolve i conflitti delle librerie Intel MKL su Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "DiemBot_Student_Project/1.0"}

# Sezioni principali del sito DIEM
DIEM_SEED_URLS = [
    "https://www.diem.unisa.it/home",
    "https://www.diem.unisa.it/dipartimento",
    "https://www.diem.unisa.it/didattica",
    "https://www.diem.unisa.it/ricerca",
    "https://www.diem.unisa.it/terza-missione",
    "https://www.diem.unisa.it/international",
]

DIEM_LABORATORI_URLS = [
    "https://www.diem.unisa.it/dipartimento/strutture?id=7",
    "https://www.diem.unisa.it/dipartimento/strutture?id=17",
    "https://www.diem.unisa.it/dipartimento/strutture?id=18",
    "https://www.diem.unisa.it/dipartimento/strutture?id=722",
    "https://www.diem.unisa.it/dipartimento/strutture?id=212",
    "https://www.diem.unisa.it/dipartimento/strutture?id=738",
    "https://www.diem.unisa.it/dipartimento/strutture?id=5",
    "https://www.diem.unisa.it/dipartimento/strutture?id=303",
    "https://www.diem.unisa.it/dipartimento/strutture?id=725",
    "https://www.diem.unisa.it/dipartimento/strutture?id=4",
    "https://www.diem.unisa.it/dipartimento/strutture?id=15",
    "https://www.diem.unisa.it/dipartimento/strutture?id=2",
    "https://www.diem.unisa.it/dipartimento/strutture?id=213",
    "https://www.diem.unisa.it/dipartimento/strutture?id=733",
    "https://www.diem.unisa.it/dipartimento/strutture?id=23",

]

MAX_DEPTH = 3
CRAWL_DELAY = 0.3  # secondi tra una richiesta e l'altra

session = requests.Session()
session.verify = False
session.headers.update(HEADERS)


def get(url: str, timeout: int = 10):
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"    [SKIP] {url} → {e}")
        return None

# ---------------------------------------------------------------------------
# PULIZIA HTML ED ESTRAZIONE LINK
# ---------------------------------------------------------------------------

def pulisci_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    for el in soup.find_all(class_=["menu", "navbar", "breadcrumb", "sidebar"]):
        el.decompose()
    testo = soup.get_text(separator="\n")
    righe = [r.strip() for r in testo.splitlines() if r.strip()]
    return "\n".join(righe)


def estrai_link_interni(html: str, base_url: str, allowed_domains: set, corsi_scoperti: set = None, allowed_prefixes: tuple = None) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        
        full = urljoin(base_url, href).split("#")[0]
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
            
        if "/en/" in parsed.path or parsed.path.endswith("/en"):
            continue
            
        domain = parsed.netloc
        clean = parsed.scheme + "://" + parsed.netloc + parsed.path
        
        if allowed_prefixes and not clean.startswith(allowed_prefixes):
            continue 
        
        if corsi_scoperti is not None and domain == "corsi.unisa.it":
            corsi_scoperti.add(clean)
            
        elif domain in allowed_domains:
            if full.lower().endswith(".pdf"):
                links.append(full)
            else:
                links.append(clean)
                
    return list(set(links))

# ---------------------------------------------------------------------------
# SCRAPER PDF E CRAWLER
# ---------------------------------------------------------------------------

def scrapa_pdf(url: str) -> Document | None:
    r = get(url)
    if r is None:
        return None
    ct = r.headers.get("Content-Type", "")
    if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
        return None
    try:
        reader = pypdf.PdfReader(io.BytesIO(r.content))
        testo = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        if testo:
            return Document(page_content=testo, metadata={"source": url, "type": "pdf"})
    except Exception as e:
        print(f"    [PDF ERROR] {url} → {e}")
    return None


def crawl(seed_urls: list[str], max_depth: int, visited: set, allowed_domains: set, corsi_scoperti: set = None, allowed_prefixes: tuple = None) -> list[Document]:
    documents = []
    queue = [(url, 0) for url in seed_urls]

    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"    [d={depth}] {url}")
        r = get(url)
        if r is None:
            continue

        final_url = r.url
        if final_url != url:
            if final_url in visited:
                continue
            visited.add(final_url)

        ct = r.headers.get("Content-Type", "")

        if "pdf" in ct.lower() or url.lower().endswith(".pdf"):
            doc = scrapa_pdf(final_url)
            if doc:
                documents.append(doc)
            time.sleep(CRAWL_DELAY)
            continue

        if "html" not in ct.lower():
            continue

        testo = pulisci_html(r.text)
        if testo:
            documents.append(Document(
                page_content=testo,
                metadata={"source": final_url, "type": "html"}
            ))

        # Passiamo i parametri aggiornati all'estrattore di link
        if depth < max_depth:
            for link in estrai_link_interni(r.text, final_url, allowed_domains, corsi_scoperti, allowed_prefixes):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(CRAWL_DELAY)

    return documents

# ---------------------------------------------------------------------------
# FASE 2: Docenti DIEM su docenti.unisa.it
# ---------------------------------------------------------------------------

def estrai_matricole_diem(url_personale: str) -> list[str]:
    r = get(url_personale)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    matricole = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "rubrica.unisa.it/persone" in href and "matricola=" in href:
            qs = parse_qs(urlparse(href).query)
            if "matricola" in qs:
                matricola = qs["matricola"][0].zfill(6)
                matricole.add(matricola)

    print(f"    Trovate {len(matricole)} matricole DIEM")
    return list(matricole)


DOCENTE_SEZIONI = ["home", "didattica"] 

def crawl_docente(matricola: str, visited: set) -> list[Document]:
    documents = []

    for sezione in DOCENTE_SEZIONI:
        url = f"https://docenti.unisa.it/{matricola}/{sezione}"
        if url in visited:
            continue
        visited.add(url)

        print(f"    [{sezione}] {url}")
        r = get(url)
        if r is None:
            continue

        if "html" not in r.headers.get("Content-Type", ""):
            continue

        testo = pulisci_html(r.text)
        if testo:
            documents.append(Document(
                page_content=testo,
                metadata={
                    "source": r.url,
                    "type": "html",
                    "matricola": matricola,
                    "sezione": sezione
                }
            ))

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"]).split("#")[0]
            if full.lower().endswith(".pdf") and full not in visited:
                visited.add(full)
                doc_pdf = scrapa_pdf(full)
                if doc_pdf:
                    documents.append(doc_pdf)

        time.sleep(CRAWL_DELAY)

    return documents

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    visited: set = set()
    tutti_i_documenti: list[Document] = []
    
    # Creiamo il cestino per raccogliere i link dei corsi
    corsi_scoperti: set = set()

    # FASE 1 – www.diem.unisa.it (Sito principale)
    print("\n=== FASE 1: Crawl www.diem.unisa.it ===")
    docs = crawl(DIEM_SEED_URLS, max_depth=MAX_DEPTH, visited=visited, allowed_domains={"www.diem.unisa.it"}, corsi_scoperti=corsi_scoperti)
    print(f"  → {len(docs)} documenti DIEM estratti")
    tutti_i_documenti.extend(docs)

    # ---------------------------------------------------------
    # NUOVA FASE 1.5 – Laboratori (Strict max_depth = 0)
    # ---------------------------------------------------------
    print("\n=== FASE 1.5: Crawl Laboratori (Depth=0) ===")
    docs_lab = crawl(DIEM_LABORATORI_URLS, max_depth=0, visited=visited, allowed_domains={"www.diem.unisa.it"})
    print(f"  → {len(docs_lab)} documenti laboratori estratti")
    tutti_i_documenti.extend(docs_lab)
    # ---------------------------------------------------------

    
    # FASE 2 – docenti.unisa.it (solo docenti DIEM)
    print("\n=== FASE 2: Docenti DIEM su docenti.unisa.it ===")
    matricole = estrai_matricole_diem("https://www.diem.unisa.it/dipartimento/personale")
    if matricole:
        docs = []
        for matricola in matricole:
            docs.extend(crawl_docente(matricola, visited))
        
        print(f"  → {len(docs)} documenti dai docenti")
        tutti_i_documenti.extend(docs)
    else:
        print("  ⚠ Nessuna matricola trovata. Controlla la pagina personale.")

    # FASE 3 – corsi.unisa.it
    print("\n=== FASE 3: Crawl DINAMICO Corsi DIEM su corsi.unisa.it ===")
    print(f"  → Trovati {len(corsi_scoperti)} link a corsi durante la Fase 1!")
    
    corsi_seeds = list(corsi_scoperti)
    
    if corsi_seeds:
        # Creiamo la "gabbia": una tupla di prefissi autorizzati
        prefissi_autorizzati = tuple(corsi_seeds)
        
        # Passiamo i prefissi al crawler
        docs = crawl(
            corsi_seeds, 
            max_depth=MAX_DEPTH, 
            visited=visited, 
            allowed_domains={"corsi.unisa.it"},
            allowed_prefixes=prefissi_autorizzati # <--- IL RECINTO!
        )
        print(f"  → {len(docs)} documenti dai corsi")
        tutti_i_documenti.extend(docs)
    else:
        print("  ⚠ Nessun link a corsi.unisa.it trovato sul sito del DIEM.")


    # Riepilogo
    print(f"\n=== TOTALE: {len(tutti_i_documenti)} documenti raccolti ===")
    conteggio: dict = {}
    for d in tutti_i_documenti:
        t = d.metadata.get("type", "?")
        conteggio[t] = conteggio.get(t, 0) + 1
    for t, n in conteggio.items():
        print(f"  {t}: {n}")

    if not tutti_i_documenti:
        print("Nessun documento! Controlla connessione e URL.")
        return

    # FASE 4 – Chunking
    print("\n=== FASE 4: Chunking ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(tutti_i_documenti)
    print(f"  → {len(chunks)} chunk totali da indicizzare")

    # FASE 5 – Embedding + Vector Store
    print("\n=== FASE 5: Creazione Vector Store (Chroma + BAAI/bge-m3 su CUDA) ===")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"}, # <-- Sfruttiamo la tua GPU nuova!
        encode_kwargs={"normalize_embeddings": True}
    )
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./diem_chroma_db"
    )
    print(f"\n✅ Fatto! Vector store salvato in './diem_chroma_db' con {len(chunks)} vettori.")


if __name__ == "__main__":
    main()