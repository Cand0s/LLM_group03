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
from langchain_huggingface import HuggingFaceEmbeddings#, embeddings

# Fixes Intel MKL library conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "DiemBot_Student_Project/1.0"}

# Main sections of the DIEM website
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
CRAWL_DELAY = 0.3  # seconds between requests

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
# HTML CLEANUP AND LINK EXTRACTION
# ---------------------------------------------------------------------------

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    for el in soup.find_all(class_=["menu", "navbar", "breadcrumb", "sidebar"]):
        el.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_internal_links(
    html: str,
    base_url: str,
    allowed_domains: set,
    discovered_course_links: set | None = None,
    allowed_prefixes: tuple | None = None,
) -> list[str]:
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
        
        if discovered_course_links is not None and domain == "corsi.unisa.it":
            discovered_course_links.add(clean)
            
        elif domain in allowed_domains:
            if full.lower().endswith(".pdf"):
                links.append(full)
            else:
                links.append(clean)
                
    return list(set(links))

# ---------------------------------------------------------------------------
# PDF SCRAPER AND CRAWLER
# ---------------------------------------------------------------------------

def scrape_pdf(url: str) -> Document | None:
    r = get(url)
    if r is None:
        return None
    ct = r.headers.get("Content-Type", "")
    if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
        return None
    try:
        reader = pypdf.PdfReader(io.BytesIO(r.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if text:
            return Document(page_content=text, metadata={"source": url, "type": "pdf"})
    except Exception as e:
        print(f"    [PDF ERROR] {url} → {e}")
    return None


def crawl(
    seed_urls: list[str],
    max_depth: int,
    visited: set,
    allowed_domains: set,
    discovered_course_links: set | None = None,
    allowed_prefixes: tuple | None = None,
) -> list[Document]:
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
            doc = scrape_pdf(final_url)
            if doc:
                documents.append(doc)
            time.sleep(CRAWL_DELAY)
            continue

        if "html" not in ct.lower():
            continue

        text = clean_html(r.text)
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": final_url, "type": "html"}
            ))

        # Pass updated parameters to the link extractor
        if depth < max_depth:
            for link in extract_internal_links(
                r.text,
                final_url,
                allowed_domains,
                discovered_course_links,
                allowed_prefixes,
            ):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(CRAWL_DELAY)

    return documents

# ---------------------------------------------------------------------------
# PHASE 2: DIEM faculty on docenti.unisa.it
# ---------------------------------------------------------------------------

def extract_diem_staff_ids(staff_page_url: str) -> list[str]:
    r = get(staff_page_url)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    staff_ids = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "rubrica.unisa.it/persone" in href and "matricola=" in href:
            qs = parse_qs(urlparse(href).query)
            if "matricola" in qs:
                staff_id = qs["matricola"][0].zfill(6)
                staff_ids.add(staff_id)

    print(f"    Found {len(staff_ids)} DIEM staff IDs")
    return list(staff_ids)


STAFF_SECTIONS = ["home", "didattica"] 

def crawl_staff_member(staff_id: str, visited: set) -> list[Document]:
    documents = []

    for section in STAFF_SECTIONS:
        url = f"https://docenti.unisa.it/{staff_id}/{section}"
        if url in visited:
            continue
        visited.add(url)

        print(f"    [{section}] {url}")
        r = get(url)
        if r is None:
            continue

        if "html" not in r.headers.get("Content-Type", ""):
            continue

        text = clean_html(r.text)
        if text:
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": r.url,
                    "type": "html",
                    "staff_id": staff_id,
                    "section": section
                }
            ))

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"]).split("#")[0]
            if full.lower().endswith(".pdf") and full not in visited:
                visited.add(full)
                doc_pdf = scrape_pdf(full)
                if doc_pdf:
                    documents.append(doc_pdf)

        time.sleep(CRAWL_DELAY)

    return documents

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    visited: set = set()
    all_documents: list[Document] = []
    
    # Collect course links discovered during crawling
    discovered_course_links: set = set()

    # PHASE 1 – www.diem.unisa.it (main website)
    print("\n=== PHASE 1: Crawl www.diem.unisa.it ===")
    docs = crawl(
        DIEM_SEED_URLS,
        max_depth=MAX_DEPTH,
        visited=visited,
        allowed_domains={"www.diem.unisa.it"},
        discovered_course_links=discovered_course_links,
    )
    print(f"  → Extracted {len(docs)} DIEM documents")
    all_documents.extend(docs)

    # ---------------------------------------------------------
    # PHASE 1.5 – Laboratories (strict max_depth = 0)
    # ---------------------------------------------------------
    print("\n=== PHASE 1.5: Crawl laboratories (depth=0) ===")
    lab_docs = crawl(DIEM_LABORATORI_URLS, max_depth=0, visited=visited, allowed_domains={"www.diem.unisa.it"})
    print(f"  → Extracted {len(lab_docs)} laboratory documents")
    all_documents.extend(lab_docs)
    # ---------------------------------------------------------
    
    # PHASE 2 – docenti.unisa.it (DIEM faculty only)
    print("\n=== PHASE 2: DIEM faculty on docenti.unisa.it ===")
    staff_ids = extract_diem_staff_ids("https://www.diem.unisa.it/dipartimento/personale")
    if staff_ids:
        docs = []
        for staff_id in staff_ids:
            docs.extend(crawl_staff_member(staff_id, visited))
        
        print(f"  → Extracted {len(docs)} faculty documents")
        all_documents.extend(docs)
    else:
        print("  ⚠ No staff IDs found. Check the staff page.")

    # PHASE 3 – corsi.unisa.it
    print("\n=== PHASE 3: Dynamic crawl of DIEM courses on corsi.unisa.it ===")
    print(f"  → Found {len(discovered_course_links)} course links during Phase 1!")
    
    course_seeds = list(discovered_course_links)
    
    if course_seeds:
        # Build the "fence": a tuple of allowed prefixes
        allowed_prefixes = tuple(course_seeds)
        
        # Pass prefixes to the crawler
        docs = crawl(
            course_seeds, 
            max_depth=MAX_DEPTH, 
            visited=visited, 
            allowed_domains={"corsi.unisa.it"},
            allowed_prefixes=allowed_prefixes  # <--- THE FENCE!
        )
        print(f"  → Extracted {len(docs)} course documents")
        all_documents.extend(docs)
    else:
        print("  ⚠ No corsi.unisa.it course links found on the DIEM website.")


    # Summary
    print(f"\n=== TOTAL: {len(all_documents)} documents collected ===")
    count_by_type: dict = {}
    for d in all_documents:
        t = d.metadata.get("type", "?")
        count_by_type[t] = count_by_type.get(t, 0) + 1
    for t, n in count_by_type.items():
        print(f"  {t}: {n}")

    if not all_documents:
        print("No documents collected! Check your connection and URLs.")
        return

    # PHASE 4 – Chunking
    print("\n=== PHASE 4: Chunking ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)
    print(f"  → {len(chunks)} total chunks to index")

    # PHASE 5 – Embedding + Vector Store
    print("\n=== PHASE 5: Build vector store (Chroma + BAAI/bge-m3 on CUDA) ===")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Running on device: {embeddings.model_kwargs['device']}")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./diem_chroma_db"
    )
    print(f"\n✅ Done! Vector store saved in './diem_chroma_db' with {len(chunks)} vectors.")


if __name__ == "__main__":
    main()