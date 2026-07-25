import chromadb
from chromadb.utils import embedding_functions
import datetime

# 1. SETUP
DB_PATH = "./legal_db"
client = chromadb.PersistentClient(path=DB_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    collection = client.get_collection(name="criminal_law", embedding_function=embedding_fn)
except:
    print("❌ ERROR: Could not find collection 'criminal_law'. Did ingest.py run?")
    exit()

# 2. CHECK COUNT
count = collection.count()
print(f"📊 Total Documents: {count}")

if count == 0:
    print("❌ DATABASE IS EMPTY. Check your JSON files and run ingest.py again.")
    exit()

# 3. PEEK AT DATA TYPES
print("\n--- 🔍 INSPECTING FIRST DOCUMENT ---")
data = collection.get(limit=1)

if data['metadatas']:
    meta = data['metadatas'][0]
    v_from = meta.get('valid_from')
    v_until = meta.get('valid_until')
    
    print(f"Doc ID: {data['ids'][0]}")
    print(f"Valid From: {v_from} (Type: {type(v_from)})")
    print(f"Valid Until: {v_until} (Type: {type(v_until)})")
    
    # 4. DIAGNOSE TYPES
    if isinstance(v_from, str):
        print("\n❌ CRITICAL ISSUE: Dates are stored as STRINGS (e.g., '20240101').")
        print("   ChromaDB requires INTEGERS (e.g., 20240101) for math filtering.")
        print("   👉 SOLUTION: Delete 'legal_db' folder and re-run the NEW ingest.py.")
    elif isinstance(v_from, int):
        print("\n✅ DATA TYPE CHECK: Dates are INTEGERS. This is good.")
        
        # 5. TEST SEARCH MANUALLY
        print("\n--- 🧪 TESTING QUERY LOGIC ---")
        test_date = 20200101 # Jan 1, 2020
        print(f"Searching for 'Theft' on date: {test_date}")
        
        results = collection.query(
            query_texts=["Theft"],
            n_results=1,
            where={
                "$and": [
                    {"valid_from": {"$lte": test_date}},
                    {"valid_until": {"$gte": test_date}}
                ]
            }
        )
        
        if results['documents'][0]:
            print("✅ SUCCESS: Query worked! The App should work.")
            print(f"Found: {results['metadatas'][0][0]['law']}")
        else:
            print("❌ FAILURE: Query returned nothing even with correct types.")
            print("   Check if your 'valid_from' / 'valid_until' ranges actually cover 2020.")
            print(f"   Example in DB: {v_from} to {v_until}")
else:
    print("❌ ERROR: Document has no metadata.")