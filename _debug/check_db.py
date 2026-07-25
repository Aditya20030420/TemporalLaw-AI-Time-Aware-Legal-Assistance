import chromadb
from chromadb.utils import embedding_functions

# 1. Connect to the database folder
chroma_client = chromadb.PersistentClient(path="./legal_db")

# 2. Setup the embedding function (Must match what you used during ingestion)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3. Get the collection
collection = chroma_client.get_collection(
    name="criminal_law",
    embedding_function=sentence_transformer_ef
)

# 4. Count total documents
count = collection.count()
print(f"✅ Total Documents in Database: {count}")

# 5. Peek at the first 3 items to verify content & metadata
print("\n--- PEEKING AT FIRST 3 DOCUMENTS ---")
data = collection.get(limit=3)

for i in range(3):
    if i < len(data['ids']):
        print(f"\n📄 ID: {data['ids'][i]}")
        print(f"   VALIDITY: {data['metadatas'][i]['valid_from']} to {data['metadatas'][i]['valid_until']}")
        print(f"   CONTENT START: {data['documents'][i][:100]}...") # Shows first 100 characters