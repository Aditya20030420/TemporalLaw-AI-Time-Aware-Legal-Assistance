# diagnostic.py - Find out what's broken

import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

DB_PATH = "./legal_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="nomic-embed-text"
)

collection = chroma_client.get_or_create_collection(
    name="criminal_law",
    embedding_function=ollama_ef
)

print("="*60)
print("DIAGNOSTIC: Checking ChromaDB for Sedition Sections")
print("="*60)

# Test 1: Check what sections exist
print("\n1. Searching for 'sedition' in database...")
results = collection.query(
    query_texts=["sedition"],
    n_results=10,
    include=["documents", "metadatas", "distances"]
)

if results['documents'][0]:
    print(f"✓ Found {len(results['documents'][0])} results")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0][:3], 
        results['metadatas'][0][:3],
        results['distances'][0][:3]
    )):
        print(f"\n  Result {i+1} (distance: {dist:.3f}):")
        print(f"  Section: {meta.get('law')} {meta.get('section')}")
        print(f"  Title: {doc[:100]}...")
else:
    print("✗ No results found for 'sedition'")

# Test 2: Check if Section 152 exists
print("\n2. Checking if BNS Section 152 (actual sedition section) exists...")
try:
    all_items = collection.get(include=["metadatas"])
    sections = [f"{m.get('law')} {m.get('section')}" for m in all_items['metadatas']]
    
    if 'BNS Section 152' in sections:
        print("✓ BNS Section 152 found in database")
    else:
        print("✗ BNS Section 152 NOT FOUND - This is why your queries fail!")
        print(f"  Database has {len(sections)} sections total")
        
        # Show what sections ARE in the DB
        print("\n  Sample sections in DB:")
        for sec in sorted(set(sections))[:20]:
            print(f"    - {sec}")
except:
    print("✗ Could not retrieve section list")

# Test 3: Check what IPC 124A returns (old sedition law)
print("\n3. Searching for 'IPC 124A' (old sedition section)...")
results = collection.query(
    query_texts=["IPC 124A sedition"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

if results['documents'][0]:
    print(f"✓ Found {len(results['documents'][0])} results")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0][:2], 
        results['metadatas'][0][:2],
        results['distances'][0][:2]
    )):
        print(f"\n  Result {i+1} (distance: {dist:.3f}):")
        print(f"  Section: {meta.get('law')} {meta.get('section')}")
else:
    print("✗ No IPC 124A found")

# Test 4: Why is Section 103(2) being returned?
print("\n4. Checking why Section 103(2) appears for sedition queries...")
results = collection.query(
    query_texts=["latest court ruling on free speech and sedition"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

print(f"Top results for sedition query:")
for i, (doc, meta, dist) in enumerate(zip(
    results['documents'][0], 
    results['metadatas'][0],
    results['distances'][0]
)):
    print(f"\n  {i+1}. {meta.get('law')} {meta.get('section')} (distance: {dist:.3f})")
    print(f"     Title: {doc.split('.')[1] if '.' in doc else doc[:80]}...")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)