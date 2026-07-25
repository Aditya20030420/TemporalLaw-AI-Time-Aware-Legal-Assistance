import streamlit as st
import requests
import os
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import ollama

# =========================
# CONFIG
# =========================
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or "388a4030633eb5f8a8a9811dd83ce8e9e3851928bae3cfc71e0f039d1ca5e363"
LEGAL_SITES = ["indiankanoon.org", "livelaw.in"]

#OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b-instruct"
#OLLAMA_MODEL = "qwen2.5:3b-instruct"

#OLLAMA_MODEL ="qwen2.5:3b-instruct-q4_K_M"
# FIX: Use a larger, more capable model
#OLLAMA_MODEL = "llama3.2:3b-instruct-q4_K_M"  # Or "mistral:7b" if available

# =========================
# CHROMA DB SETUP
# =========================
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

# =========================
# ROUTING LOGIC
# =========================
def requires_web_search(query: str, offline_hits: int) -> bool:
    keyword_triggers = [
        "recent", "latest", "judgment", "bail",
        "supreme court", "high court",
        "status", "stayed", "constitutional"
    ]

    if offline_hits == 0:
        return True

    return any(k in query.lower() for k in keyword_triggers)

# =========================
# SERPAPI SEARCH
# =========================
def serpapi_search(query: str, year: int):
    site_filter = " OR ".join([f"site:{s}" for s in LEGAL_SITES])
    extra_terms = ""
    extra_terms = ""
    if any(word in query.lower() for word in ["judgment", "judgement", "case", "court", "supreme", "high court", "bench"]):
        extra_terms = "judgment OR order OR decision OR ruling"

    final_query = f"{query} {extra_terms} {year} {site_filter}".strip()

    params = {
        "engine": "google",
        "q": final_query,
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_KEY,
        "num": 5
    }

    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=15)
        data = r.json()
    except Exception:
        return []

    results = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title", "Untitled"),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })
    return results

def filter_recent_results(results, year: int):
    filtered = []
    for r in results:
        text = (r.get("snippet") or "").lower()
        if str(year) in text or "recent" in text or "latest" in text:
            filtered.append(r)
    return filtered if filtered else results  # Return all if none match year

# =========================
# CHROMA SEARCH (TIME-AWARE) - FIXED
# =========================
def chroma_search(query: str, query_date: datetime, top_k: int = 5):
    """
    FIX: Use semantic similarity from ChromaDB embeddings instead of keyword matching.
    ChromaDB already ranks results by semantic relevance.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 2,  # Get more candidates for time filtering
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    query_int = int(query_date.strftime("%Y%m%d"))

    for doc, meta, distance in zip(
        results["documents"][0], 
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Time-based filtering
        if not (meta["valid_from"] <= query_int <= meta["valid_until"]):
            continue

        # FIX: Trust semantic similarity (lower distance = more relevant)
        # Only include if reasonably relevant (distance < 1.0 is a good threshold)
        if distance < 0.8:
            hits.append({
                "title": f"{meta['law']} Section {meta['section']}",
                "text": doc,
                "source": meta.get("source", "Offline DB"),
                "relevance": 1 - distance  # Convert to relevance score
            })

        if len(hits) >= top_k:
            break

    # Sort by relevance
    hits.sort(key=lambda x: x["relevance"], reverse=True)
    return hits

# =========================
# CONTEXT BUILDERS
# =========================
def build_chroma_context(results):
    if not results:
        return ["No applicable statutory provisions found for the selected date."]

    blocks = []
    for r in results:
        blocks.append(
            f"STATUTE:\n{r['title']}\n{r['text']}\nSource: {r['source']}"
        )
    return blocks
def build_web_context(results):
    if not results:
        return ["No recent judicial decisions found from trusted legal sources."]
    
    blocks = []
    for r in results:
        # Extract date if in snippet (simple heuristic; improve with regex if needed)
        date_str = ""
        snippet_lower = r['snippet'].lower()
        possible_dates = [word for word in snippet_lower.split() if word.isdigit() and len(word) == 4 and 2000 <= int(word) <= 2030]
        if possible_dates:
            date_str = f" (Reported: {possible_dates[0]})"  # crude; better parsing later
        
        blocks.append(
            f"JUDICIAL UPDATE -> \n{r['title']}{date_str}\n{r['snippet']}\nLink: {r['link']}"
        )
    return blocks

# =========================
# LLM ANSWER GENERATION - IMPROVED
# =========================
def generate_answer(query: str, context_blocks: list, has_web_results: bool):
    priority_instruction = ""
    if has_web_results and "judgment" in query.lower():
        priority_instruction = """

options={
    "temperature": 0.3,
    "top_p": 0.9,
    "num_ctx": 4096,       # Limit context to avoid extra RAM spike
    "num_predict": 512     # Limit max output tokens
}    
CRITICAL:
This query concerns judicial decisions.
Prioritize JUDICIAL UPDATE sections.
Do not invent case law.
"""
    fallback_instruction = ""
    if not context_blocks:
        fallback_instruction = """
If no context is available, clearly state that no authoritative sources were found,
but explain the settled legal position where possible.
"""

    #system_prompt = f"""
#You are a legal assistant for Indian law.
#Answer strictly from the provided context.
#{priority_instruction}
#{fallback_instruction}
#""".strip()
    
    system_prompt = f"""
You are a precise Indian legal assistant. Base EVERY statement strictly on the provided Context blocks ONLY.
- Quote key phrases/titles/links accurately from JUDICIAL UPDATE blocks.
- For judgment queries: Clearly state the court (e.g., High Court vs Supreme Court), date, holding, and outcome (e.g., bail granted, sedition not made out).
- Do NOT speculate, hedge excessively, or introduce unrelated topics like defamation unless in context.
- If context is directly on-point, explain the ruling plainly — do not say "did not address" if it does.
- If mismatch, say: "The retrieved source is from [court] and holds [summary]. It does not involve Supreme Court clarification on [topic]."
{priority_instruction}
{fallback_instruction}
""".strip()
    
    user_prompt = f"""
Query:
{query}
Context:
{chr(10).join(context_blocks)}
""".strip()

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.3,  # Slightly higher for better generation without hallucination
                "top_p": 0.9
            }
        )
        # Log raw response for debugging
        print("OLLAMA response:", response)
        return response['message']['content'].strip()
    except Exception as e:
        return f"LLM unavailable: {str(e)}"
# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Time-Aware Legal Assistant", layout="wide")
st.title("Time-Aware Legal Assistant (IPC ↔ BNS)")

query = st.text_input("Enter legal query")
selected_date = st.date_input("Select relevant date", value=datetime.today())

if st.button("Analyze") and query:
    with st.spinner("Analyzing legal sources..."):
        chroma_results = chroma_search(query, selected_date)
        chroma_context = build_chroma_context(chroma_results)

        use_web = requires_web_search(query, len(chroma_results))
        web_context = []
        if use_web:
            raw_web = serpapi_search(query, selected_date.year)
            web_results = filter_recent_results(raw_web, selected_date.year)
            web_context = build_web_context(web_results)
        else:
            web_context = ["Web search not required."]


        # FIX: For judicial queries, prioritize web context
      
        # Always put more authoritative/recent info first
        if "judgment" in query.lower() or "court" in query.lower() or "case" in query.lower():
            combined_context = web_context + chroma_context
        else:
            combined_context = chroma_context + web_context

        has_web_results = any("JUDICIAL UPDATE" in block for block in web_context)

        answer = generate_answer(query, combined_context, has_web_results)

    st.markdown("### Final Answer")
    st.write(answer)

    with st.expander("Offline Statutory Provisions"):
        for c in chroma_context:
            st.text(c)
            st.write("---")

    with st.expander("Judicial & Web Sources"):
        for w in web_context:
            st.text(w)
            st.write("---")