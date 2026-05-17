"""
chunk_inspector.py
==================
Script di analisi della qualità dei chunk (parent e children) prodotti da indexer.py.

UTILIZZO
--------
    # Analisi completa (legge parent_store.json + child ChromaDB)
    python chunk_inspector.py

    # Analisi del solo file di debug export (debug_parent_child_chunks.json)
    python chunk_inspector.py --debug-only

    # Esporta report HTML interattivo
    python chunk_inspector.py --html

DIPENDENZE
----------
    pip install langchain-chroma langchain-core rich pandas matplotlib

Le stesse dipendenze di indexer.py sono sufficienti, più `rich` e `pandas`.
"""

import json
import argparse
import statistics
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────
# CONFIGURAZIONE  (deve corrispondere a indexer.py)
# ──────────────────────────────────────────────────────────────
PARENT_STORE_PATH      = "./diem_chroma_db/parent_store.json"
CHILD_DB_DIRECTORY     = "./diem_chroma_db/children"
CHILD_COLLECTION_NAME  = "diem_children"
DEBUG_EXPORT_PATH      = "./debug_parent_child_chunks.json"
CHILD_DB_READ_BATCH_SIZE = 500

# Soglie di qualità (modificabili)
THRESHOLDS = {
    # ── Parent ──────────────────────────────────────────────
    "parent_min_chars":        100,    # Un parent troppo corto è quasi sicuramente rumore
    "parent_max_chars":       2000,    # Sopra questa soglia il contesto per l'LLM diventa pesante
    "parent_ideal_min":        400,    # Range ideale per un buon contesto
    "parent_ideal_max":       1600,

    # ── Child ───────────────────────────────────────────────
    "child_min_chars":          80,    # Child troppo corti degradano il retrieval
    "child_max_chars":         700,    # Child troppo lunghi "annegano" il segnale semantico
    "child_ideal_min":         150,
    "child_ideal_max":         550,

    # ── Struttura ───────────────────────────────────────────
    "max_children_per_parent":  10,    # Troppi child per parent = parent troppo grande
    "min_children_per_parent":   1,    # Deve esserci almeno 1 child per parent
    "max_overlap_ratio":        0.4,   # Overlap tra child consecutivi: soglia massima

    # ── Contenuto ───────────────────────────────────────────
    "min_word_count":            8,    # Chunk quasi vuoti (solo punteggiatura/numeri)
    "max_repetition_ratio":     0.6,   # Rapporto parole uniche / totali: sotto = molto ripetitivo
    "min_sentence_count":        1,    # Almeno una frase compiuta
}

# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class ChunkIssue:
    severity: str          # "error" | "warning" | "info"
    code: str              # identificatore breve
    message: str

@dataclass
class ChunkReport:
    chunk_id: str
    chunk_type: str        # "parent" | "child"
    source: str
    length: int
    word_count: int
    sentence_count: int
    has_header: bool
    has_section: bool
    repetition_ratio: float
    issues: List[ChunkIssue] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Punteggio qualità 0-100 (100 = perfetto)."""
        deductions = sum(
            15 if i.severity == "error" else 5
            for i in self.issues
        )
        return max(0, 100 - deductions)

    @property
    def quality_label(self) -> str:
        s = self.score
        if s >= 85: return "✅ Ottimo"
        if s >= 65: return "⚠️  Discreto"
        if s >= 40: return "🔶 Scarso"
        return "❌ Critico"


# ──────────────────────────────────────────────────────────────
# HELPERS DI ANALISI DEL TESTO
# ──────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())

def _sentence_count(text: str) -> int:
    """Stima approssimativa del numero di frasi."""
    return max(1, len(re.split(r'[.!?]+', text)))

def _repetition_ratio(text: str) -> float:
    """
    Rapporto parole uniche / totale parole.
    Valori bassi (< 0.4) indicano testo molto ripetitivo o boilerplate.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def _has_markdown_header(text: str) -> bool:
    return bool(re.search(r'^#{1,5}\s+\S', text, re.MULTILINE))

def _has_section_breadcrumb(text: str) -> bool:
    return "Section:" in text

def _overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Stima la sovrapposizione testuale tra due chunk adiacenti
    basandosi sulle ultime/prime N parole.
    """
    words_a = text_a.split()[-80:]
    words_b = text_b.split()[:80]
    if not words_a or not words_b:
        return 0.0
    set_a, set_b = set(words_a), set(words_b)
    return len(set_a & set_b) / min(len(set_a), len(set_b))


# ──────────────────────────────────────────────────────────────
# ANALISI CHUNK
# ──────────────────────────────────────────────────────────────

def analyze_chunk(chunk_id: str, text: str, chunk_type: str,
                  metadata: Dict) -> ChunkReport:
    T = THRESHOLDS
    length   = len(text)
    words    = _word_count(text)
    sents    = _sentence_count(text)
    rep      = _repetition_ratio(text)
    has_hdr  = _has_markdown_header(text)
    has_sec  = _has_section_breadcrumb(text)
    source   = metadata.get("source", "unknown")

    issues: List[ChunkIssue] = []

    # ── Lunghezza ────────────────────────────────────────────
    min_c = T[f"{chunk_type}_min_chars"]
    max_c = T[f"{chunk_type}_max_chars"]
    ideal_min = T[f"{chunk_type}_ideal_min"]
    ideal_max = T[f"{chunk_type}_ideal_max"]

    if length < min_c:
        issues.append(ChunkIssue("error", "TOO_SHORT",
            f"Lunghezza {length} < {min_c} chars: chunk quasi vuoto."))
    elif length < ideal_min:
        issues.append(ChunkIssue("warning", "BELOW_IDEAL",
            f"Lunghezza {length} è sotto il range ideale ({ideal_min}-{ideal_max})."))
    elif length > max_c:
        issues.append(ChunkIssue("error", "TOO_LONG",
            f"Lunghezza {length} > {max_c} chars: potrebbe degradare la qualità del retrieval/contesto."))
    elif length > ideal_max:
        issues.append(ChunkIssue("warning", "ABOVE_IDEAL",
            f"Lunghezza {length} supera il range ideale ({ideal_min}-{ideal_max})."))

    # ── Contenuto minimo ─────────────────────────────────────
    if words < T["min_word_count"]:
        issues.append(ChunkIssue("error", "FEW_WORDS",
            f"Solo {words} parole: probabile rumore o frammento."))

    if sents < T["min_sentence_count"]:
        issues.append(ChunkIssue("warning", "NO_SENTENCE",
            "Nessuna frase riconoscibile: il chunk potrebbe essere solo metadati/header."))

    # ── Ripetitività ─────────────────────────────────────────
    if rep < T["max_repetition_ratio"]:
        issues.append(ChunkIssue("warning", "REPETITIVE",
            f"Ratio parole uniche={rep:.2f}: testo molto ripetitivo o boilerplate."))

    # ── Metadati ─────────────────────────────────────────────
    if chunk_type == "child":
        if not metadata.get("parent_id"):
            issues.append(ChunkIssue("error", "NO_PARENT_ID",
                "child_id presente ma parent_id mancante: link interrotto."))
        if not metadata.get("child_id"):
            issues.append(ChunkIssue("error", "NO_CHILD_ID",
                "child_id mancante: impossibile fare lookup al retrieval."))
        if not has_hdr and not has_sec:
            issues.append(ChunkIssue("info", "NO_CONTEXT_INJECTION",
                "Nessun header Markdown né breadcrumb 'Section:' rilevato nel testo."))

    if chunk_type == "parent":
        if not metadata.get("parent_id"):
            issues.append(ChunkIssue("error", "NO_PARENT_ID",
                "parent_id mancante: impossibile fare lookup al retrieval."))

    return ChunkReport(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        source=source,
        length=length,
        word_count=words,
        sentence_count=sents,
        has_header=has_hdr,
        has_section=has_sec,
        repetition_ratio=rep,
        issues=issues,
    )


# ──────────────────────────────────────────────────────────────
# CARICAMENTO DATI
# ──────────────────────────────────────────────────────────────

def load_parent_store(path: str) -> Dict[str, Dict]:
    """Carica il ParentStore JSON su disco."""
    p = Path(path)
    if not p.exists():
        print(f"⚠️  ParentStore non trovato: {path}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_child_db_raw(directory: str, collection: str) -> List[Dict]:
    """
    Legge i child direttamente da ChromaDB senza embedding model.
    Restituisce lista di dict con keys: id, document, metadata.
    """
    try:
        import chromadb
    except ImportError:
        print("⚠️  chromadb non installato. Installa con: pip install chromadb")
        return []

    p = Path(directory)
    if not p.exists():
        print(f"⚠️  Child DB non trovato: {directory}")
        return []

    client = chromadb.PersistentClient(path=str(p))
    try:
        col = client.get_collection(collection)
    except Exception as e:
        print(f"⚠️  Collezione '{collection}' non trovata: {e}")
        return []

    chunks: List[Dict] = []

    try:
        total = col.count()
    except Exception:
        total = None

    # Chroma su SQLite puo' fallire con "too many SQL variables" quando si usa get() su collezioni grandi.
    # Leggiamo quindi a blocchi tramite limit/offset.
    batch_size = CHILD_DB_READ_BATCH_SIZE
    offset = 0
    while True:
        try:
            result = col.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
        except Exception as e:
            err = str(e).lower()
            if "too many sql variables" in err and batch_size > 50:
                print("⚠️  Query troppo grande per SQLite, riduco il batch size e riprovo...")
                # fallback conservativo se anche 500 fosse troppo alto su certi ambienti
                batch_size = 100
                continue
            raise

        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        if not ids:
            break

        for cid, doc, meta in zip(ids, docs, metas):
            chunks.append({"id": cid, "document": doc, "metadata": meta or {}})

        offset += len(ids)
        if total is not None and offset >= total:
            break

    return chunks

def load_debug_export(path: str) -> List[Dict]:
    """Carica il file debug_parent_child_chunks.json di indexer.py."""
    p = Path(path)
    if not p.exists():
        print(f"⚠️  Debug export non trovato: {path}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# REPORT AGGREGATO
# ──────────────────────────────────────────────────────────────

def _stats(values: List[float]) -> Dict:
    if not values:
        return {"count": 0}
    return {
        "count":  len(values),
        "min":    round(min(values), 1),
        "max":    round(max(values), 1),
        "mean":   round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "stdev":  round(statistics.stdev(values), 1) if len(values) > 1 else 0,
    }

def aggregate_reports(reports: List[ChunkReport]) -> Dict:
    """Calcola statistiche aggregate su una lista di ChunkReport."""
    lengths = [r.length for r in reports]
    scores  = [r.score for r in reports]

    issue_counter: Counter = Counter()
    for r in reports:
        for iss in r.issues:
            issue_counter[f"{iss.severity.upper()}:{iss.code}"] += 1

    errors   = sum(1 for r in reports for i in r.issues if i.severity == "error")
    warnings = sum(1 for r in reports for i in r.issues if i.severity == "warning")

    quality_dist = Counter(r.quality_label for r in reports)
    sources      = Counter(r.source for r in reports)

    return {
        "total":          len(reports),
        "length_stats":   _stats(lengths),
        "score_stats":    _stats(scores),
        "errors":         errors,
        "warnings":       warnings,
        "top_issues":     issue_counter.most_common(10),
        "quality_dist":   dict(quality_dist),
        "top_sources":    sources.most_common(10),
    }


# ──────────────────────────────────────────────────────────────
# ANALISI STRUTTURALE PARENT↔CHILD
# ──────────────────────────────────────────────────────────────

def analyze_parent_child_structure(
    parents: Dict[str, Dict],
    children: List[Dict]
) -> Dict:
    """
    Verifica l'integrità della struttura parent-child:
    - Ogni child ha un parent?
    - Ogni parent ha almeno un child?
    - La distribuzione children-per-parent è sana?
    """
    T = THRESHOLDS
    child_by_parent: Dict[str, List] = defaultdict(list)
    orphan_children = []

    for c in children:
        pid = c["metadata"].get("parent_id")
        if pid and pid in parents:
            child_by_parent[pid].append(c)
        else:
            orphan_children.append(c["id"])

    parents_without_children = [pid for pid in parents if pid not in child_by_parent]
    children_per_parent = [len(v) for v in child_by_parent.values()]

    # Overlap tra child consecutivi
    overlap_ratios = []
    for pid, clist in child_by_parent.items():
        docs = [c["document"] for c in clist]
        for i in range(len(docs) - 1):
            r = _overlap_ratio(docs[i], docs[i + 1])
            overlap_ratios.append(r)

    high_overlap = sum(1 for r in overlap_ratios if r > T["max_overlap_ratio"])

    return {
        "total_parents":            len(parents),
        "total_children":           len(children),
        "orphan_children":          len(orphan_children),
        "parents_without_children": len(parents_without_children),
        "children_per_parent":      _stats(children_per_parent),
        "high_overlap_pairs":       high_overlap,
        "total_consecutive_pairs":  len(overlap_ratios),
        "overlap_stats":            _stats(overlap_ratios),
        "parents_without_children_ids": parents_without_children[:10],  # Campione
    }


# ──────────────────────────────────────────────────────────────
# STAMPA RICH (con fallback plain-text)
# ──────────────────────────────────────────────────────────────

def _try_rich():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import print as rprint
        return Console(), Table, Panel, rprint
    except ImportError:
        return None, None, None, None

def print_section(title: str, console=None):
    sep = "═" * 60
    if console:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n{sep}\n  {title}\n{sep}")

def print_kv(label: str, value, console=None):
    if console:
        console.print(f"  [bold]{label:<40}[/bold] {value}")
    else:
        print(f"  {label:<40} {value}")


def print_full_report(
    parent_agg: Dict,
    child_agg: Dict,
    structure: Dict,
    worst_parents: List[ChunkReport],
    worst_children: List[ChunkReport],
    sample_pairs: List[Tuple],
):
    console, Table, Panel, rprint = _try_rich()

    # ── Riepilogo Generale ───────────────────────────────────
    print_section("📊  RIEPILOGO GENERALE", console)
    print_kv("Parent chunks totali",  parent_agg["total"], console)
    print_kv("Child chunks totali",   child_agg["total"], console)
    ratio = child_agg["total"] / max(parent_agg["total"], 1)
    print_kv("Ratio child/parent",    f"{ratio:.2f}", console)

    # ── Statistiche Parent ───────────────────────────────────
    print_section("📦  PARENT CHUNKS — statistiche", console)
    ls = parent_agg["length_stats"]
    print_kv("Lunghezza min/media/max (chars)",
             f"{ls['min']} / {ls['mean']} / {ls['max']}", console)
    print_kv("Deviazione std lunghezza", ls.get("stdev", "N/A"), console)
    ss = parent_agg["score_stats"]
    print_kv("Score qualità min/media/max",
             f"{ss['min']} / {ss['mean']} / {ss['max']}", console)
    print_kv("Chunk con errori (almeno 1)", parent_agg["errors"], console)
    print_kv("Chunk con warning (almeno 1)", parent_agg["warnings"], console)
    print("\n  Distribuzione qualità:")
    for label, cnt in sorted(parent_agg["quality_dist"].items()):
        pct = 100 * cnt / max(parent_agg["total"], 1)
        print(f"    {label:<25} {cnt:>6}  ({pct:.1f}%)")

    print("\n  Top issues riscontrate:")
    for code, cnt in parent_agg["top_issues"]:
        print(f"    {code:<35} {cnt:>5}×")

    # ── Statistiche Child ────────────────────────────────────
    print_section("🔍  CHILD CHUNKS — statistiche", console)
    ls = child_agg["length_stats"]
    print_kv("Lunghezza min/media/max (chars)",
             f"{ls['min']} / {ls['mean']} / {ls['max']}", console)
    ss = child_agg["score_stats"]
    print_kv("Score qualità min/media/max",
             f"{ss['min']} / {ss['mean']} / {ss['max']}", console)
    print_kv("Chunk con errori", child_agg["errors"], console)
    print_kv("Chunk con warning", child_agg["warnings"], console)
    print("\n  Distribuzione qualità:")
    for label, cnt in sorted(child_agg["quality_dist"].items()):
        pct = 100 * cnt / max(child_agg["total"], 1)
        print(f"    {label:<25} {cnt:>6}  ({pct:.1f}%)")

    print("\n  Top issues riscontrate:")
    for code, cnt in child_agg["top_issues"]:
        print(f"    {code:<35} {cnt:>5}×")

    # ── Struttura Parent↔Child ───────────────────────────────
    print_section("🔗  STRUTTURA PARENT ↔ CHILD", console)
    print_kv("Child orfani (parent_id mancante o non trovato)",
             structure["orphan_children"], console)
    print_kv("Parent senza child",
             structure["parents_without_children"], console)
    cpp = structure["children_per_parent"]
    print_kv("Child/parent  min/media/max",
             f"{cpp.get('min','?')} / {cpp.get('mean','?')} / {cpp.get('max','?')}", console)
    ov = structure["overlap_stats"]
    if ov.get("count", 0) > 0:
        print_kv("Overlap consecutivo medio",
                 f"{ov['mean']:.2f} (max {ov['max']:.2f})", console)
    print_kv("Coppie con overlap eccessivo (> soglia)",
             f"{structure['high_overlap_pairs']} / {structure['total_consecutive_pairs']}", console)

    if structure["parents_without_children_ids"]:
        print("\n  Campione parent_id senza children:")
        for pid in structure["parents_without_children_ids"][:5]:
            print(f"    …{pid[-12:]}")

    # ── Chunk Peggiori ───────────────────────────────────────
    if worst_parents:
        print_section("❌  TOP 5 PARENT PEGGIORI", console)
        for r in worst_parents[:5]:
            print(f"\n  [{r.quality_label}]  score={r.score}  len={r.length}")
            print(f"  source: {r.source}")
            print(f"  id: …{r.chunk_id[-12:]}")
            for iss in r.issues:
                sym = "✖" if iss.severity == "error" else "⚠"
                print(f"    {sym} [{iss.code}] {iss.message}")

    if worst_children:
        print_section("❌  TOP 5 CHILD PEGGIORI", console)
        for r in worst_children[:5]:
            print(f"\n  [{r.quality_label}]  score={r.score}  len={r.length}")
            print(f"  source: {r.source}")
            print(f"  id: …{r.chunk_id[-12:]}")
            for iss in r.issues:
                sym = "✖" if iss.severity == "error" else "⚠"
                print(f"    {sym} [{iss.code}] {iss.message}")

    # ── Coppie di Esempio ────────────────────────────────────
    if sample_pairs:
        print_section("🔎  COPPIE PARENT↔CHILD DI ESEMPIO", console)
        for i, (parent_text, children_texts, meta) in enumerate(sample_pairs[:3]):
            print(f"\n  ── Coppia #{i+1} ── source: {meta.get('source','?')} ──")
            print(f"  PARENT ({len(parent_text)} chars):")
            preview = parent_text[:400].replace("\n", " ")
            print(f"    {preview}{'…' if len(parent_text) > 400 else ''}")
            for j, ct in enumerate(children_texts[:3]):
                print(f"\n  CHILD #{j+1} ({len(ct)} chars):")
                cp = ct[:200].replace("\n", " ")
                print(f"    {cp}{'…' if len(ct) > 200 else ''}")
            if len(children_texts) > 3:
                print(f"    … (+{len(children_texts)-3} altri child)")

    # ── Linee Guida Finali ───────────────────────────────────
    print_section("📋  INTERPRETAZIONE DEI RISULTATI", console)
    print(_evaluation_guide())


def _evaluation_guide() -> str:
    return """
  GUIDA ALLA VALUTAZIONE DEI CHUNK PER AGENTIC RAG
  ─────────────────────────────────────────────────

  1. LUNGHEZZA
     • Parent ideale: 400–1600 chars → contesto ricco ma non eccessivo per l'LLM.
     • Child ideale:  150–550 chars  → granularità giusta per il retrieval semantico.
     • Parent troppo corti (<100): rischio di perder contesto; valuta aumentare parent_size.
     • Child troppo corti (<80):   embedding debole; aumenta child_size o revisa i separatori.
     • Child troppo lunghi (>700): il retrieval "centra" meno; riduci child_size.

  2. STRUTTURA PARENT↔CHILD
     • Ogni parent DEVE avere almeno 1 child → orphan_children = 0 è l'obiettivo.
     • Parent senza child = parent mai recuperabili → fix urgente.
     • Child/parent ideale: 2–5. Sopra 8 il parent è probabilmente troppo grande.
     • Overlap eccessivo (>40%) tra child consecutivi → riduci child_overlap.

  3. CONTENUTO
     • Repetition ratio basso (<0.4) su molti chunk → potresti avere boilerplate
       (footer, navigazione HTML) che "inquina" il corpus. Filtra a monte nel JSON.
     • Chunk con poche parole (<8) → quasi certamente frammenti di UI o rumore.

  4. ARRICCHIMENTO CONTESTUALE (context injection)
     • I child devono iniziare con "Document: … / Section: …":
       verifica che la flag NO_CONTEXT_INJECTION non sia alta.
     • Parent devono iniziare con "Document: …": controlla la struttura.

  5. SCORE AGGREGATO
     • Score medio > 85: configurazione buona.
     • Score medio 65–85: alcune regolazioni consigliate (vedi top issues).
     • Score medio < 65: revisione della configurazione chunk necessaria.

  AZIONI CORRETTIVE TIPICHE
  ─────────────────────────
  TOO_SHORT (parent): aumenta parent_size o abbassa i separatori markdown.
  TOO_SHORT (child):  aumenta child_size; controlla se i documenti sorgenti
                      sono già molto brevi (tipo FAQ a risposta corta).
  REPETITIVE:         aggiungi un filtro nel loader per rimuovere sezioni di
                      navigazione/boilerplate dal JSON sorgente.
  ORPHAN_CHILDREN:    verifica che parent_store.json sia aggiornato e che
                      l'hash deterministic sia stabile tra i run.
  HIGH_OVERLAP:       riduci child_overlap (es. da 50 a 20 chars).
"""


# ──────────────────────────────────────────────────────────────
# EXPORT HTML OPZIONALE
# ──────────────────────────────────────────────────────────────

def export_html_report(
    parent_reports: List[ChunkReport],
    child_reports: List[ChunkReport],
    structure: Dict,
    output_path: str = "chunk_quality_report.html"
):
    """Genera un report HTML statico con tabelle e grafici base."""

    def issue_html(issues):
        if not issues:
            return "<span style='color:green'>—</span>"
        parts = []
        for i in issues:
            col = "#c0392b" if i.severity == "error" else "#e67e22" if i.severity == "warning" else "#2980b9"
            parts.append(f"<span style='color:{col}'>[{i.code}] {i.message}</span>")
        return "<br>".join(parts)

    def rows_html(reports, limit=200):
        rows = []
        for r in sorted(reports, key=lambda x: x.score)[:limit]:
            bg = "#fde8e8" if r.score < 50 else "#fff8e1" if r.score < 75 else "#e8f8e8"
            rows.append(f"""
            <tr style="background:{bg}">
                <td style='font-size:10px'>{r.chunk_id[-12:]}</td>
                <td>{r.source[:40]}</td>
                <td>{r.length}</td>
                <td>{r.word_count}</td>
                <td>{r.score}</td>
                <td>{r.quality_label}</td>
                <td>{issue_html(r.issues)}</td>
            </tr>""")
        return "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Chunk Quality Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 30px; background: #f5f5f5; }}
  h1, h2 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; }}
  th {{ background: #2c3e50; color: white; padding: 8px; text-align: left; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; font-size: 13px; }}
  .stat {{ display: inline-block; background: white; border-radius: 8px;
           padding: 12px 20px; margin: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .stat .val {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
</style>
</head>
<body>
<h1>🔍 Chunk Quality Report — Agentic RAG</h1>

<h2>Riepilogo Struttura</h2>
<div>
  <div class='stat'><div class='val'>{structure['total_parents']}</div>Parent chunks</div>
  <div class='stat'><div class='val'>{structure['total_children']}</div>Child chunks</div>
  <div class='stat'><div class='val'>{structure['orphan_children']}</div>Orphan children</div>
  <div class='stat'><div class='val'>{structure['parents_without_children']}</div>Parent senza child</div>
  <div class='stat'><div class='val'>{structure['children_per_parent'].get('mean','?')}</div>Child/parent (media)</div>
</div>

<h2>Parent Chunks — peggiori (score crescente)</h2>
<table>
<tr><th>ID (ultimi 12)</th><th>Source</th><th>Length</th><th>Words</th><th>Score</th><th>Qualità</th><th>Issues</th></tr>
{rows_html(parent_reports)}
</table>

<h2>Child Chunks — peggiori (score crescente)</h2>
<table>
<tr><th>ID (ultimi 12)</th><th>Source</th><th>Length</th><th>Words</th><th>Score</th><th>Qualità</th><th>Issues</th></tr>
{rows_html(child_reports)}
</table>

</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Report HTML salvato in: {output_path}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analizzatore qualità chunk RAG")
    parser.add_argument("--debug-only", action="store_true",
                        help="Usa solo debug_parent_child_chunks.json (non richiede ChromaDB)")
    parser.add_argument("--html", action="store_true",
                        help="Esporta anche un report HTML interattivo")
    parser.add_argument("--parent-store", default=PARENT_STORE_PATH)
    parser.add_argument("--child-db",     default=CHILD_DB_DIRECTORY)
    parser.add_argument("--debug-export", default=DEBUG_EXPORT_PATH)
    parser.add_argument("--html-out",     default="chunk_quality_report.html")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHUNK QUALITY INSPECTOR  —  Agentic RAG")
    print("=" * 60)

    # ── Caricamento ──────────────────────────────────────────
    if args.debug_only:
        print(f"\n[MODE] Debug-only: leggo {args.debug_export}")
        debug_data = load_debug_export(args.debug_export)
        if not debug_data:
            print("Nessun dato trovato. Esegui prima indexer.py con DEBUG_EXPORT=True.")
            return

        # Ricostruiamo liste sintetiche di parent e child dal debug export
        parents_raw = {}
        children_raw = []
        for entry in debug_data:
            pid = entry["parent_id"]
            parents_raw[pid] = {
                "page_content": entry.get("parent_preview", ""),
                "metadata": {"source": entry.get("source","?"),
                             "type":   entry.get("type","?"),
                             "parent_id": pid},
            }
            for c in entry.get("children", []):
                children_raw.append({
                    "id": c.get("child_id","?"),
                    "document": c.get("preview",""),
                    "metadata": {"parent_id": pid,
                                 "child_id": c.get("child_id","?"),
                                 "source": entry.get("source","?")},
                })
    else:
        print(f"\n[MODE] Full: leggo ParentStore + ChromaDB")
        parents_raw  = load_parent_store(args.parent_store)
        children_raw = load_child_db_raw(args.child_db, CHILD_COLLECTION_NAME)

    if not parents_raw and not children_raw:
        print("\n⚠️  Nessun dato disponibile. Verifica i path e riesegui indexer.py.")
        return

    print(f"\n  Caricati {len(parents_raw)} parent e {len(children_raw)} child.")

    # ── Analisi ──────────────────────────────────────────────
    print("\n  Analisi parent in corso…")
    parent_reports: List[ChunkReport] = []
    for pid, pdata in parents_raw.items():
        text = pdata.get("page_content", "")
        meta = pdata.get("metadata", {})
        meta["parent_id"] = pid
        r = analyze_chunk(pid, text, "parent", meta)
        parent_reports.append(r)

    print("  Analisi child in corso…")
    child_reports: List[ChunkReport] = []
    for cdata in children_raw:
        cid  = cdata["id"]
        text = cdata["document"]
        meta = cdata["metadata"]
        r = analyze_chunk(cid, text, "child", meta)
        child_reports.append(r)

    # ── Aggregazione ─────────────────────────────────────────
    parent_agg = aggregate_reports(parent_reports)
    child_agg  = aggregate_reports(child_reports)
    structure  = analyze_parent_child_structure(parents_raw, children_raw)

    # ── Coppie di esempio (3 casuali) ────────────────────────
    child_by_parent: Dict[str, List] = defaultdict(list)
    for c in children_raw:
        pid = c["metadata"].get("parent_id")
        if pid:
            child_by_parent[pid].append(c["document"])

    sample_pairs = []
    sample_pids  = list(parents_raw.keys())[:3]
    for pid in sample_pids:
        pdata = parents_raw[pid]
        sample_pairs.append((
            pdata.get("page_content",""),
            child_by_parent.get(pid, []),
            pdata.get("metadata", {}),
        ))

    # ── Stampa ───────────────────────────────────────────────
    worst_parents  = sorted(parent_reports, key=lambda r: r.score)[:5]
    worst_children = sorted(child_reports,  key=lambda r: r.score)[:5]

    print_full_report(
        parent_agg, child_agg, structure,
        worst_parents, worst_children, sample_pairs
    )

    # ── Export HTML ──────────────────────────────────────────
    if args.html:
        export_html_report(parent_reports, child_reports, structure, args.html_out)

    # ── Export JSON riepilogo ────────────────────────────────
    summary = {
        "parent_summary": parent_agg,
        "child_summary":  child_agg,
        "structure":      structure,
    }
    out = Path("chunk_quality_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Riepilogo JSON salvato in: {out}")


if __name__ == "__main__":
    main()