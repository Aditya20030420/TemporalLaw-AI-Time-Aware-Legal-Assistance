"""
ChromaDB Database Population Script
====================================
Embeds every IPC + BNS section (from the JSON files) into ChromaDB using the
Ollama `nomic-embed-text` model, so the semantic retrieval path in app.py can
find all sections.

Run once after changing the JSON corpus:
    python populate_database.py

Requires:
    - Ollama running (ollama serve) with: ollama pull nomic-embed-text
    - IPC_updated.json and BNS_updated.json present
"""

import json
import chromadb
import requests
from pathlib import Path

# ----- CONFIG (must match app.py) -----
DB_PATH = "./legal_db"
COLLECTION = "criminal_law"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
IPC_JSON_PATH = "IPC_updated.json"
BNS_JSON_PATH = "BNS_updated.json"


def embed(text: str):
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def load(path, law):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  loaded {len(data)} {law} sections from {path}")
    return data


def doc_text(sec):
    """Rich text used for the embedding (mirrors the MiniLM index in app.py)."""
    return (
        f"{sec.get('law','')} Section {sec.get('section','')} "
        f"{sec.get('title','')} {sec.get('category','')} "
        f"{sec.get('content','') or sec.get('text','')} {sec.get('punishment','')}"
    ).strip()


def main():
    print("=" * 60)
    print("POPULATING CHROMADB (nomic-embed-text)")
    print("=" * 60)

    sections = load(IPC_JSON_PATH, "IPC") + load(BNS_JSON_PATH, "BNS")
    print(f"  total sections: {len(sections)}")

    client = chromadb.PersistentClient(path=DB_PATH)

    # Wipe any stale collection so the store exactly matches the JSON corpus.
    try:
        client.delete_collection(COLLECTION)
        print("  deleted existing collection")
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION)

    added = errors = 0
    total = len(sections)
    for i, sec in enumerate(sections, 1):
        doc_id = f"{sec['law']}_{sec['section']}"
        try:
            collection.add(
                ids=[doc_id],
                embeddings=[embed(doc_text(sec))],
                documents=[doc_text(sec)],
                metadatas=[{
                    "law": sec.get("law", ""),
                    "section": str(sec.get("section", "")),
                    "title": sec.get("title", ""),
                    "category": sec.get("category", ""),
                    "valid_from": str(sec.get("valid_from", "")),
                    "valid_until": str(sec.get("valid_until", "")),
                }],
            )
            added += 1
        except Exception as e:
            errors += 1
            print(f"  [{i}/{total}] error on {doc_id}: {str(e)[:80]}")
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] embedded...")

    print(f"\nDone. added={added} errors={errors} total_in_db={collection.count()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure Ollama is running and `ollama pull nomic-embed-text` is done.")
