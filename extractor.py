"""
DIEM Chatbot - Web Scraper 
"""

import os
import ssl
import tempfile
import time
import json
from urllib.parse import urlparse, parse_qs, urljoin
import re

import urllib3
import requests
from bs4 import BeautifulSoup
import trafilatura
import pymupdf4llm

from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# ---------------------------------------------------------------------------
# CONFIGURATION & GLOBAL STATE
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MAX_DEPTH_DIEM = 4
MAX_DEPTH_COURSES = 3
MIN_COURSE_YEAR = 2021      # corsi con anno < MIN_COURSE_YEAR vengono scartati
FORBIDDEN_EXTENSIONS = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".ttf", ".json")

PROCESSED_URLS = set()
DISCOVERED_COURSES = set()
DISCOVERED_STRUCTURES = set()
DISCOVERED_PDF = set()

# ---------------------------------------------------------------------------
# SSL CONFIGURATION
# ---------------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

original_get = requests.get

def unverified_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)

requests.get = unverified_get

# ---------------------------------------------------------------------------
# EXTRACTION HELPERS
# ---------------------------------------------------------------------------
def extract_diem_prof_ids(staff_page_url: str) -> list[str]:
    r = requests.get(staff_page_url, verify=False)
    if r.status_code != 200:
        return []
    
    soup = BeautifulSoup(r.text, "html.parser")
    staff_ids = set()
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "rubrica.unisa.it/persone" in href and "matricola=" in href:
            qs = parse_qs(urlparse(href).query)
            if "matricola" in qs:
                staff_ids.add(qs["matricola"][0].zfill(6))
                
    return list(staff_ids)



def clean_html(raw_html: str, url: str = "") -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["header", "nav"]):
        tag.decompose()
    footers = soup.find_all(
        lambda t: t.name in ("footer",) or
                  (t.name == "div" and t.get("id") in ("footer", "subfooter")) or
                  (t.name == "div" and "footer" in " ".join(t.get("class", [])))
    )
    for footer in footers:
        for sibling in footer.find_next_siblings():
            sibling.decompose()
        footer.decompose()

    result = trafilatura.extract(
        str(soup),
        url=url,         
        output_format="markdown",          
        include_tables=True,
        include_links=False,       
        include_images=False,
        include_comments=False,    # Includi eventuali sezioni commenti o note a margine
        favor_precision=True,    # Più selettivo: scarta sezioni considerate "non centrali" (nav, sidebar, ads)
        deduplicate=True,        # Disattiva la rimozione dei duplicati interni alla pagina
    )


    return result.strip() if result else ""





def print_website_metadata(raw_html: str, url: str) -> dict:
    parsed_url = urlparse(url)
    qs = parse_qs(parsed_url.query)
    
    for key, values in qs.items():
        if "struttura" in key.lower():
            if "300638" not in values:
                print(f"      [SKIPPED - Wrong Department ({values[0]})] {url}")
                return {"source": "discard_wrong_department"}

    url_base = url.split("?")[0].lower() 
    if url_base.endswith(FORBIDDEN_EXTENSIONS):
        print(f"      [SKIPPED - CSS/Media] {url}")
        return {"source": "discard_extension"}
        
    # FIX: use canonical URL form (https, lowercase host, no trailing slash)
    # to avoid treating http:// and https://, or trailing-slash variants, as different pages.
    url_pulito = _normalize_url(url)
    if url_pulito in PROCESSED_URLS:
        print(f"      [SKIPPED - Duplicate] {url}")
        return {"source": "discard_duplicate"}

    PROCESSED_URLS.add(url_pulito)
    
    print(f"      [SUCCESS] {url}")
    soup = BeautifulSoup(raw_html, "html.parser")
    
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(url, href).split("#")[0]

        # FIX: discard any link that carries an explicit anno= parameter older than MIN_COURSE_YEAR.
        # Applies to corsi.unisa.it, docenti.unisa.it/didattica, and any other URL with anno=.
        qs_anno = parse_qs(urlparse(full_url).query).get("anno", [None])[0]
        if qs_anno is not None:
            try:
                if int(qs_anno) < MIN_COURSE_YEAR:
                    continue
            except ValueError:
                pass  # unparseable anno — let it through

        if full_url.lower().endswith(".pdf"):
            DISCOVERED_PDF.add(full_url)
        elif "corsi.unisa.it" in full_url:
            DISCOVERED_COURSES.add(full_url)
        elif "diem.unisa.it/dipartimento/strutture" in full_url and "id=" in full_url:
            DISCOVERED_STRUCTURES.add(full_url)

    return {"source": url}

# ---------------------------------------------------------------------------
# CRAWLING PHASES
# ---------------------------------------------------------------------------
def crawl_diem_base() -> list[Document]:
    print("\n=== PHASE 1: Crawl www.diem.unisa.it ===")
    docs_collected = []
    print("  Crawling seed: https://www.diem.unisa.it/")
    loader = RecursiveUrlLoader(
        url="https://www.diem.unisa.it/",
        max_depth=MAX_DEPTH_DIEM,
        extractor=clean_html, 
        metadata_extractor=print_website_metadata,
        exclude_dirs=["https://www.diem.unisa.it/en", "https://www.diem.unisa.it/en/"],
        prevent_outside=True
    )
    for doc in loader.load():
        source_url = doc.metadata.get("source", "").lower()
        
        if source_url.startswith("discard"):
            continue
 
        doc.metadata["type"] = "html"
        docs_collected.append(doc)
                
    print(f"  -> Extracted {len(docs_collected)} DIEM documents")
    return docs_collected



def crawl_structures() -> list[Document]:
    print(f"\n=== PHASE 1.1: Strutture DIEM ({len(DISCOVERED_STRUCTURES)} pagine) ===")
    docs_collected = []
    
    for url in DISCOVERED_STRUCTURES:
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code != 200:
                continue
            
            content = clean_html(r.text, url=url)
            if not content or not content.strip():
                print(f"      [SKIPPED - Empty] {url}")
                continue
                
            docs_collected.append(Document(
                page_content=content,
                metadata={"source": url, "type": "html"}
            ))
            print(f"      [SUCCESS] {url}")
            
        except Exception as e:
            print(f"      [SKIPPED] {url} - {e}")

        time.sleep(0.5)
        
    return docs_collected



def crawl_professor_courses() -> list[Document]:
    print("\n=== PHASE 2: DIEM Professors on docenti.unisa.it ===")
    docs_collected = []
    staff_ids = extract_diem_prof_ids("https://www.diem.unisa.it/dipartimento/personale")

    if not staff_ids:
        return docs_collected

    staff_urls = []
    for staff_id in staff_ids:
        staff_urls.append(f"https://docenti.unisa.it/{staff_id}/home")
        staff_urls.append(f"https://docenti.unisa.it/{staff_id}/didattica")

    for url in staff_urls:
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # ── /home ────────────────────────────────────────────────────
            if url.endswith("/home"):
                if not soup.find("table"):
                    print(f"      [SKIPPED - No home page found] {url}")
                    continue
                content = clean_html(r.text, url=url)
                if not content or not content.strip():
                    print(f"      [SKIPPED - Empty] {url}")
                    continue
                docs_collected.append(Document(
                    page_content=content,
                    metadata={"source": url, "type": "html"}
                ))
                print(f"      [SUCCESS] {url}")

            # ── /didattica ───────────────────────────────────────────────
            # Non salva la pagina principale (solo codici, niente nomi corso)
            # Salva SOLO i dettagli dei singoli corsi cliccabili
            elif url.endswith("/didattica"):
                course_links = set()
                skipped_old = 0
                for a in soup.find_all("a", href=True):
                    full = urljoin(url, a["href"]).split("#")[0]
                    if "didattica?anno=" in full and "id=" in full:
                        # Discard courses older than MIN_COURSE_YEAR
                        qs_anno = parse_qs(urlparse(full).query).get("anno", ["0"])[0]
                        try:
                            anno = int(qs_anno)
                        except ValueError:
                            anno = 0
                        if anno < MIN_COURSE_YEAR:
                            skipped_old += 1
                            continue
                        course_links.add(full)

                if skipped_old:
                    print(f"      [FILTER] {skipped_old} corsi scartati (anno < {MIN_COURSE_YEAR})")

                if not course_links:
                    print(f"      [SKIPPED - No courses found] {url}")
                    continue

                print(f"      [DIDATTICA] {url} → {len(course_links)} corsi")
                for course_url in course_links:
                    try:
                        rc = requests.get(course_url, verify=False, timeout=15)
                        if rc.status_code != 200:
                            continue
                        course_content = clean_html(rc.text, url=course_url)
                        if not course_content or not course_content.strip():
                            continue
                        docs_collected.append(Document(
                            page_content=course_content,
                            metadata={"source": course_url, "type": "html"}
                        ))
                        print(f"         [COURSE] {course_url}")
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"         [SKIPPED course] {course_url} - {e}")

        except Exception as e:
            print(f"      [SKIPPED] {url} - {e}")
        time.sleep(0.5)

    print(f"  -> Extracted {len(docs_collected)} professor documents")
    return docs_collected



def crawl_courses() -> list[Document]:
    print("\n=== PHASE 3: DIEM courses ===")
    docs_collected = []
    course_seeds = list(DISCOVERED_COURSES)
    
    if course_seeds:
        print(f"  -> Trovati {len(course_seeds)} link a corsi durante la Fase 1!")
        for url in course_seeds:
            course_loader = RecursiveUrlLoader(
                url=url,
                max_depth=MAX_DEPTH_COURSES, 
                extractor=clean_html,
                metadata_extractor=print_website_metadata,
                prevent_outside=True
            )
            docs = course_loader.load()
            
            for doc in docs:
                source_url = doc.metadata.get("source", "").lower()
                
                # Scarta le pagine classificate come discard (es. CSS, JS)
                if source_url.startswith("discard"):
                    continue

                # Scarta le pagine in inglese
                if "/en/" in source_url or source_url.endswith("/en"):
                    print(f"      [SKIPPED - English] {source_url}")
                    continue
               
                # Se arriva fin qui, è una pagina valida! Aggiungi il tipo e salvala.
                doc.metadata["type"] = "html"
                docs_collected.append(doc)  # <-- Usa append qui!
                
        print("  -> Corsi estratti correttamente.")
    else:
        print("  Nessun link ai corsi trovato.")
        
    return docs_collected





def download_pdfs() -> list[Document]:
    print(f"\n=== PHASE 4: Download {len(DISCOVERED_PDF)} PDF intercepted ===")
    docs_collected = []

    for pdf_url in DISCOVERED_PDF:
        try:
            r = requests.get(pdf_url, verify=False, timeout=20)
            if r.status_code != 200:
                print(f"      [SKIPPED] {pdf_url}")
                continue

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            # Genera il Markdown
            md = pymupdf4llm.to_markdown(tmp_path)

            if md:
                # --- NOVITÀ: Pulisci i placeholder delle immagini ---
                # Cerca la stringa esatta e rimuovila, assorbendo anche gli eventuali "a capo" (\s*) extra
                md = re.sub(r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*\s*", "", md)

                # Mantieni la tua logica di sostituzione dei separatori
                md = md.replace('\u2028', '\n').replace('\u2029', '\n\n')
                
            os.unlink(tmp_path)

            if not md or not md.strip():
                print(f"      [VUOTO] {pdf_url}")
                continue

            docs_collected.append(Document(
                page_content=md,
                metadata={"source": pdf_url, "type": "pdf"}
            ))
            print(f"      [OK PDF] {pdf_url}")

        except Exception as e:
            print(f"      [SKIPPED] {pdf_url} — {e}")

    return docs_collected




# ---------------------------------------------------------------------------
# DATA PROCESSING & EXPORT
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    """
    Canonical form of a URL for deduplication.
    Normalizes scheme to https, lowercases host, strips trailing slash.
    e.g. http://www.diem.unisa.it/home/ -> https://www.diem.unisa.it/home
    """
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    normalized = parsed._replace(scheme="https", netloc=parsed.netloc.lower())
    return normalized.geturl()


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Two-pass deduplication:
      Pass 1 - URL dedup: keeps only the first document for each canonical URL
               (handles trailing slash, http/https, lowercase host differences).
      Pass 2 - Content hash dedup: removes documents with identical page_content
               regardless of URL. This is the main fix for pages like / and /home
               that are served with different URLs but identical HTML.
    """
    import hashlib
    print("\n=== CLEANING DATA: removing duplicates ===")
    before = len(documents)

    # Pass 1: URL normalization dedup
    by_url: dict[str, Document] = {}
    for doc in documents:
        key = _normalize_url(doc.metadata.get("source", ""))
        if key not in by_url:
            by_url[key] = doc
    after_url = len(by_url)

    # Pass 2: content hash dedup
    seen_hashes: set[str] = set()
    unique: list[Document] = []
    for doc in by_url.values():
        h = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)

    print(f"  Documenti in input:       {before}")
    print(f"  Dopo dedup URL:           {after_url}  (-{before - after_url})")
    print(f"  Dopo dedup contenuto:     {len(unique)}  (-{after_url - len(unique)})")
    return unique

# Minimum character threshold below which a document is considered degenerate
# (navigation-only pages, empty sections, menu fragments with no real content).
MIN_CONTENT_CHARS = 150


def export_to_json(documents: list[Document], filename: str):
    """
    Export documents to JSON, filtering out degenerate entries.
    Documents with page_content shorter than MIN_CONTENT_CHARS are skipped:
    they are typically nav menus, breadcrumbs, or empty sections that become
    useless micro-chunks in the indexer and pollute the vector store.
    """
    print(f"\n=== EXPORTING DATA: saving to {filename} ===")
    dataset_finale = []
    skipped_short = 0

    for doc in documents:
        text = doc.page_content
        if len(text) < MIN_CONTENT_CHARS:
            skipped_short += 1
            continue
        dataset_finale.append({
            "text": text,
            "source": doc.metadata.get("source", ""),
            "type": doc.metadata.get("type", "sconosciuto")
        })

    if skipped_short:
        print(f"  [FILTER] Skipped {skipped_short} documents shorter than {MIN_CONTENT_CHARS} chars (nav/menu/empty).")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset_finale, f, indent=4, ensure_ascii=False)

    print(f"  Data saved to '{filename}' with {len(dataset_finale)} entries.")

def print_summary(documents: list[Document]):
    print(f"\n=== TOTAL: {len(documents)} unique documents collected ===")
    if not documents:
        print("No documents collected! Check your connection and URLs.")
        return
    
    count_pdf = 0
    count_html = 0
    dettaglio_tipi = {}

    for doc in documents:
        tipo = doc.metadata.get("type", "sconosciuto")
        if "pdf" in tipo:
            count_pdf += 1
        elif "html" in tipo:
            count_html += 1
            
        dettaglio_tipi[tipo] = dettaglio_tipi.get(tipo, 0) + 1

    print(f"  -> Totale Pagine HTML: {count_html}")
    print(f"  -> Totale File PDF:    {count_pdf}")
    print("  -> Dettaglio per fase di estrazione:")
    
    for tipo, quantita in dettaglio_tipi.items():
        print(f"      - {tipo}: {quantita}")

    print("\nFASE DI INGESTIONE COMPLETATA!")
    print("Ora puoi eseguire 'python indexer.py' per creare il Vector Store.")







# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    all_documents: list[Document] = []

    all_documents.extend(crawl_diem_base())
    all_documents.extend(crawl_structures())
    all_documents.extend(crawl_professor_courses())
    all_documents.extend(crawl_courses())
    all_documents.extend(download_pdfs())

    all_documents = deduplicate_documents(all_documents)
    
    export_to_json(all_documents, "diem_knowledge_base.json")
    print_summary(all_documents)

if __name__ == "__main__":
    main()