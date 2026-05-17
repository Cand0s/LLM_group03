"""kb_stats.py

Print useful statistics about the JSON knowledge base.

Main output:
- file size on disk
- number of records
- distribution by content type
- statistics about text lengths
- most frequent sources and source domains
- presence of missing fields

Usage:
    python kb_stats.py
    python kb_stats.py --path diem_knowledge_base.json
"""

from __future__ import annotations
import os
import json
import argparse
from collections import Counter, defaultdict
from urllib.parse import urlparse
from statistics import mean, median

def normalize_type(value: object) -> str:
    if value is None:
        return "missing"
    text = str(value).strip().lower()
    return text or "missing"

def source_domain(source: object) -> str:
    if not source:
        return "missing"
    parsed = urlparse(str(source))
    if not parsed.netloc:
        return "local_or_unknown"
    return parsed.netloc.lower()

def safe_len(value: object) -> int:
    return len(value) if isinstance(value, str) else 0

def print_section(title: str) -> None:
    print(f"\n=== {title} ===")

def main() -> None:
    parser = argparse.ArgumentParser(description="Print statistics for the JSON knowledge base")
    parser.add_argument("--path", default="diem_knowledge_base.json", help="Path to the JSON file")
    parser.add_argument("--top", type=int, default=10, help="Number of items to show in leaderboards")
    args = parser.parse_args()

    json_path = args.path
    top_n = args.top

    file_size = os.path.getsize(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The JSON file must contain a list of records")

    total_records = len(data)
    
    # Extract texts and calculate general lengths
    texts = [entry.get("text", "") for entry in data if isinstance(entry, dict)]
    lengths = [safe_len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts if isinstance(text, str)]

    # Group lengths by document type to calculate specific medians
    lengths_by_type = defaultdict(list)
    for entry in data:
        if isinstance(entry, dict):
            doc_type = normalize_type(entry.get("type"))
            doc_length = safe_len(entry.get("text"))
            lengths_by_type[doc_type].append(doc_length)

    types = Counter(normalize_type(entry.get("type")) for entry in data if isinstance(entry, dict))
    domains = Counter(source_domain(entry.get("source")) for entry in data if isinstance(entry, dict))
    sources = Counter(str(entry.get("source", "missing")) for entry in data if isinstance(entry, dict))

    missing_text = sum(1 for entry in data if not isinstance(entry, dict) or not entry.get("text"))
    missing_source = sum(1 for entry in data if not isinstance(entry, dict) or not entry.get("source"))
    missing_type = sum(1 for entry in data if not isinstance(entry, dict) or not entry.get("type"))

    print_section("Overview")
    print(f"File: {json_path}")
    print(f"File size on disk: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"Total records: {total_records}")
    print(f"Valid records (dict): {sum(1 for entry in data if isinstance(entry, dict))}")

    print_section("Missing fields")
    print(f"Missing text:   {missing_text}")
    print(f"Missing source: {missing_source}")
    print(f"Missing type:   {missing_type}")

    print_section("Distribution by type")
    for type_name, count in types.most_common():
        print(f"{type_name:20} {count:8}  ({count / total_records * 100:.2f}%)")

    print_section("Text statistics")
    if lengths:
        print(f"Average characters: {mean(lengths):.1f}")
        print(f"Min characters:     {min(lengths)}")
        print(f"Max characters:     {max(lengths)}")
        
        # Replaced general median with median by type
        print("\nMedian characters by type:")
        for doc_type, type_lengths in sorted(lengths_by_type.items()):
            if type_lengths:
                print(f"  - {doc_type:10}: {median(type_lengths):.1f}")

    if word_counts:
        print(f"\nAverage words:      {mean(word_counts):.1f}")
        print(f"Median words:       {median(word_counts):.1f}")

    print_section(f"Top {top_n} sources")
    for source, count in sources.most_common(top_n):
        print(f"{count:8}  {source}")

    print_section("Source domains")
    for domain, count in domains.most_common(top_n):
        print(f"{count:8}  {domain}")

    print_section(f"Top {top_n} longest records")
    longest = sorted(
        (
            (safe_len(entry.get("text")), str(entry.get("source", "missing")), normalize_type(entry.get("type")))
            for entry in data
            if isinstance(entry, dict)
        ),
        reverse=True,
    )[:top_n]
    for length, source, type_name in longest:
        print(f"{length:8} chars | {type_name:10} | {source}")


if __name__ == "__main__":
    main()