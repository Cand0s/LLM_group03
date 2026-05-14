"""
DIEM Chatbot - Web Scraper & Vector Store Builder
"""

import os
import ssl
import time
import json
import logging
from urllib.parse import urlparse, parse_qs, urljoin
from sympy import re
import urllib3
import requests
from bs4 import BeautifulSoup
import trafilatura
from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------------------------------
# CONFIGURATION & GLOBAL STATE
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MAX_DEPTH_DIEM = 4
MAX_DEPTH_COURSES = 3
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
    """Override requests.get to disable SSL verification by default."""
    kwargs['verify'] = False
    return original_get(*args, **kwargs)

requests.get = unverified_get

# ---------------------------------------------------------------------------
# EXTRACTION HELPERS
# ---------------------------------------------------------------------------
def extract_diem_prof_ids(staff_page_url: str) -> list[str]:
    """Extract staff IDs from the DIEM staff page.

    Args:
        staff_page_url (str): URL of the DIEM staff page.

    Returns:
        list[str]: List of staff IDs (matricola) extracted from the page."""
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
    """Clean raw HTML by removing headers/footers/nav and extracting the main text.

    Args:
        raw_html (str): The raw HTML content to be cleaned.
        url (str): The URL of the page (optional).

    Returns:
        str: The cleaned main text."""
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
        include_links=True,
        include_images=False,
        include_comments=False,  # Include any comment sections or marginal notes
        favor_precision=True,    # Disable the surgical trimming of parts considered "non central"
        deduplicate=True,        # Disable removal of duplicates within the page
    )

    return result.strip() if result else ""

def print_website_metadata(raw_html: str, url: str) -> dict:
    """
    Extract metadata from the URL and raw HTML to determine if the page should be processed or skipped.

    Args:
        raw_html (str): The raw HTML content of the page.
        url (str): The URL of the page.

    Returns:
        dict: A dictionary containing metadata about the page, including whether it should be processed or skipped.
    """
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

    url_pulito = url.rstrip("/")
    if url_pulito in PROCESSED_URLS:
        print(f"      [SKIPPED - Duplicate] {url}")
        return {"source": "discard_duplicate"}

    PROCESSED_URLS.add(url_pulito)

    print(f"      [SUCCESS] {url}")
    soup = BeautifulSoup(raw_html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(url, href).split("#")[0]

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
    """
    Crawl the main DIEM website starting from the homepage, extracting HTML content while applying
    filters to skip irrelevant pages.

    Returns:
        list[Document]: A list of Document objects containing the extracted content and metadata.
    """

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
    """
    Crawl the DIEM structures pages discovered during the initial crawl, extracting their content.
    Returns:
        list[Document]: A list of Document objects containing the extracted content and metadata.
    """
    print(f"\n=== PHASE 1.1: DIEM Structures ({len(DISCOVERED_STRUCTURES)} pages) ===")
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

def crawl_faculty() -> list[Document]:
    """
    Crawl the DIEM faculty pages on docenti.unisa.it, starting from the staff IDs extracted from the DIEM staff page.
    For each professor, extract both the /home and /didattica pages, and from the latter also follow links to individual course details.
    Returns:
        list[Document]: A list of Document objects containing the extracted content and metadata.
    """
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

            if url.endswith("/didattica") and not soup.find("table"):
                print(f"      [SKIPPED - No courses found] {url}")
                continue
            if url.endswith("/home") and not soup.find("table"):
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

            # === DEPTH 2: single course details ===
            # Only from /didattica pages, follow didattica?anno=X&id=X links
            if url.endswith("/didattica"):
                course_links = set()
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    full = urljoin(url, href).split("#")[0]
                    if "didattica?anno=" in full and "id=" in full:
                        course_links.add(full)

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

    print(f"  -> Extracted {len(docs_collected)} faculty documents")
    return docs_collected



def crawl_courses() -> list[Document]:
    """
    Crawl the course pages on corsi.unisa.it, starting from the course links discovered during the previous phases.
    For each course, extract the main content while skipping English pages and applying the same cleaning as before.
    Returns:
        list[Document]: A list of Document objects containing the extracted content and metadata.
    """
    print("\n=== PHASE 3: DIEM courses ===")
    docs_collected = []
    course_seeds = list(DISCOVERED_COURSES)

    if course_seeds:
        print(f"  -> Found {len(course_seeds)} course links during Phase 1!")
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

                if "/en/" in source_url or source_url.endswith("/en"):
                    print(f"      [SKIPPED - English] {source_url}")
                    continue
                doc.metadata["type"] = "html"
            docs_collected.extend(docs)
        print("  -> Courses extracted successfully.")
    else:
        print("  No course links found.")

    return docs_collected

def download_pdfs() -> list[Document]:
    """
    Download and extract text from all PDF URLs discovered during the crawling phases,
    merging multi-page PDFs into single documents.

    Returns:
        list[Document]: A list of Document objects containing the extracted content and metadata from the PDFs.
    """
    print(f"\n=== PHASE 4: Download {len(DISCOVERED_PDF)} PDF intercepted ===")
    docs_collected = []
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    
    for pdf_url in list(DISCOVERED_PDF):
        try:
            pdf_loader = PyPDFLoader(pdf_url)
            pages = pdf_loader.load()

            # Merge the text from all pages so the deduplicator does not remove them
            testo_completo = "\n\n".join([p.page_content for p in pages])

            if testo_completo.strip():
                merged_doc = Document(
                    page_content=testo_completo,
                    metadata={"source": pdf_url, "type": "pdf"}
                )
                docs_collected.append(merged_doc)
                print(f"      [OK PDF] {pdf_url} ({len(pages)} merged pages)")
            else:
                print(f"      [EMPTY] PDF without decodable text: {pdf_url}")

        except Exception as e:
            print(f"      [SKIPPED] PDF error: {pdf_url} — {e}")

    return docs_collected


# ---------------------------------------------------------------------------
# DATA PROCESSING & EXPORT
# ---------------------------------------------------------------------------
def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Remove duplicate documents based on their source URL, keeping only one document per unique URL.

    Args:
        documents (list[Document]): A list of Document objects to deduplicate.

    Returns:
        list[Document]: A list of unique Document objects.
    """
    print("\n=== CLEANING DATA: removing duplicates ===")
    documenti_unici = {}

    for doc in documents:
        url_pulito = doc.metadata.get("source", "").rstrip("/")
        if url_pulito not in documenti_unici:
            documenti_unici[url_pulito] = doc

    return list(documenti_unici.values())


def export_to_json(documents: list[Document], filename: str):
    """
    Export the list of Document objects to a JSON file, including their text content and metadata.

    Args:
        documents (list[Document]): A list of Document objects to export.
        filename (str): The name of the JSON file to create.
    """
    print(f"\n=== EXPORTING DATA: saving to {filename} ===")
    dataset_finale = []

    for doc in documents:
        dataset_finale.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "type": doc.metadata.get("type", "sconosciuto")
        })

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

    print(f"  -> Total HTML pages: {count_html}")
    print(f"  -> Total PDF files:    {count_pdf}")
    print("  -> Breakdown by extraction phase:")

    for tipo, quantita in dettaglio_tipi.items():
        print(f"      - {tipo}: {quantita}")

    print("\nINGESTION PHASE COMPLETED!")
    print("You can now run 'python indexer.py' to create the Vector Store.")

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    all_documents: list[Document] = []

    all_documents.extend(crawl_diem_base())         # Phase 1: crawl the main DIEM website
    all_documents.extend(crawl_structures())        # Phase 1.1: crawl the discovered structures pages for more content
    all_documents.extend(crawl_faculty())           # Phase 2: crawl the DIEM faculty pages on docenti.unisa.it, extracting both home and didattica pages
    all_documents.extend(crawl_courses())           # Phase 3: crawl the course pages on corsi.unisa.it, starting from the discovered course links and applying the same cleaning
    all_documents.extend(download_pdfs())           # Phase 4: download and extract text from all discovered PDF URLs, merging multi-page PDFs into single documents

    all_documents = deduplicate_documents(all_documents)  # Remove duplicates based on source URL, keeping only one document per unique URL

    export_to_json(all_documents, "diem_knowledge_base.json")  # Export final documents to JSON for indexing and chatbot use
    print_summary(all_documents)  # Print summary with counts by type and extraction phase

if __name__ == "__main__":
    main()