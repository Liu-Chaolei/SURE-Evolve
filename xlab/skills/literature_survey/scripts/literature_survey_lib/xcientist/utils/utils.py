import hashlib
import re, json
from typing import Dict
import os
import pypdfium2 as pdfium

def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _escape_invalid_backslashes(text: str) -> str:
    chars = []
    in_string = False
    escaped = False
    i = 0
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}

    while i < len(text):
        ch = text[i]
        if not in_string:
            chars.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
            i += 1
            continue

        if escaped:
            if ch not in valid_escapes:
                chars.append("\\")
            chars.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\":
            chars.append(ch)
            escaped = True
            i += 1
            continue

        chars.append(ch)
        if ch == '"':
            in_string = False
        i += 1

    return "".join(chars)


def _replace_unquoted_literals(text: str) -> str:
    chars = []
    in_string = False
    escaped = False
    i = 0
    replacements = {
        "None": "null",
        "True": "true",
        "False": "false",
        "NaN": "null",
        "Infinity": "null",
        "-Infinity": "null",
    }
    keys = sorted(replacements.keys(), key=len, reverse=True)

    while i < len(text):
        ch = text[i]
        if in_string:
            chars.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            chars.append(ch)
            i += 1
            continue

        matched = False
        for token in keys:
            if text.startswith(token, i):
                prev = text[i - 1] if i > 0 else ""
                nxt = text[i + len(token)] if i + len(token) < len(text) else ""
                if not (prev.isalnum() or prev == "_") and not (nxt.isalnum() or nxt == "_"):
                    chars.append(replacements[token])
                    i += len(token)
                    matched = True
                    break
        if matched:
            continue

        chars.append(ch)
        i += 1

    return "".join(chars)


def _repair_json_like_text(text: str) -> str:
    repaired = text
    repaired = _escape_invalid_backslashes(repaired)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = _replace_unquoted_literals(repaired)
    return repaired


def extract_json(text):
    # remove ```json fences
    if not text:
        raise ValueError("Empty text in json extraction function")
    text = re.sub(r"```[\w]*", "", text).replace("```", "")
    text = text.strip()

    # Decode one complete root value; harmless trailing prose is not part of JSON.
    # Never search inside a truncated root for a smaller, misleading valid object.
    m = re.search(r"[\[{]", text)
    if not m:
        raise ValueError("No JSON found")
    candidate = text[m.start():]
    try:
        return json.JSONDecoder().raw_decode(candidate)[0]
    except json.JSONDecodeError:
        repaired = _repair_json_like_text(candidate)
        return json.JSONDecoder().raw_decode(repaired)[0]


def is_valid_pdf(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 2048:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(5)
            size = os.path.getsize(path)
            f.seek(max(size - 20, 0), os.SEEK_SET)
            tail = f.read()
        if not (head == b"%PDF-" and b"%%EOF" in tail):
            return False
        pdfium.PdfDocument(path)
        return True
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return False


if __name__ == "__main__":
    json_data = """```json
    {
        "cluster_name": "Taxonomy and Structural Frameworks for Knowledge Organization",
        "summary": "This cluster includes methodologies for automatic taxonomy generation and knowledge organization, focusing on structured frameworks to enhance navigation and usefulness of scholarly content.",
        "papers": [
            {
                "id": "2510.17263",
                "title": "TAXOALIGN: Automating Scholarly Taxonomy Generation",
                "tldr": "The paper presents TAXOALIGN, an innovative approach for automating the generation of scholarly taxonomies using large language models, significantly outperforming existing methods in structural alignment and semantic coherence."
            }
        ]
    }
```"""
    data = extract_json(json_data)
    for cluster in data:
        print("Cluster Name:", cluster["cluster_name"])
        print("Summary:", cluster["summary"])
        print("Papers:")
        for paper in cluster["papers"]:
            print(f"  - ID: {paper['id']}")
            print(f"    Title: {paper['title']}")
            print(f"    TL;DR: {paper['tldr']}")
        print()
