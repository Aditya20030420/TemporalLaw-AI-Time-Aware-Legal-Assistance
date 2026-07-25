import streamlit as st
import requests
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or "388a4030633eb5f8a8a9811dd83ce8e9e3851928bae3cfc71e0f039d1ca5e363"

LEGAL_SITES = [
    "indiankanoon.org",
    "livelaw.in"
]

# =========================
# ROUTER
# =========================
def requires_web_search(query):
    triggers = [
        "recent", "latest", "judgment",
        "supreme court", "high court",
        "stayed", "constitutional",
        "amendment", "status"
    ]
    return any(t in query.lower() for t in triggers)

# =========================
# SERPAPI SEARCH
# =========================
def serpapi_search(query):
    site_filter = " OR ".join([f"site:{s}" for s in LEGAL_SITES])
    current_year = datetime.now().year
    final_query = f"{query} Supreme Court judgment {current_year} {site_filter}"


    params = {
        "engine": "google",
        "q": final_query,
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_KEY,
        "num": 5
    }

    r = requests.get("https://serpapi.com/search", params=params, timeout=15)
    data = r.json()

    results = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })

    return results

# =========================
# OFFLINE STATUTE MOCK
# =========================

def offline_statute_lookup(query, selected_date):
    if "sedition" in query.lower():
        return {
            "law": "IPC Section 124A",
            "text": "Sedition: Whoever brings or attempts to bring into hatred or contempt, or excites disaffection...",
            "valid_from": "1860",
            "valid_until": "2022"
        }
    return None

# =========================
# CONTEXT BUILDERS
# =========================
def build_web_context(results):
    if not results:
        return "No recent judicial updates found from web search."

    lines = []
    for r in results:
        if r["snippet"]:
            lines.append(
                f"- {r['title']}: {r['snippet']} (Source: {r['link']})"
            )
    return "\n".join(lines)

def build_offline_context(statute):
    if not statute:
        return "No relevant offline statute found."

    return (
        f"Statute: {statute['law']}\n"
        f"Text: {statute['text']}\n"
        f"Validity: {statute['valid_from']} – {statute['valid_until']}"
    )

# =========================
# ANSWER GENERATION (LLM PLACEHOLDER)
# =========================
def generate_answer(query, web_ctx, offline_ctx):
    if "sedition" in query.lower() and "abeyance" in web_ctx.lower():
        return (
            "The Supreme Court of India, in 2022, kept the operation of the "
            "sedition law (IPC Section 124A) in abeyance while the government "
            "reconsidered its constitutional validity."
        )

    return (
        "Based on the available legal information, the answer has been derived "
        "from statutory provisions and publicly available judicial summaries."
    )

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Time-Aware Legal Assistant", layout="wide")

st.title("🕰️ Time-Aware Legal Assistance (IPC ↔ BNS)")

query = st.text_input("Enter Legal Query")
selected_date = st.date_input("Select Date", value=datetime.today())

if st.button("Analyze") and query:
    st.subheader("Legal Analysis")

    use_web = requires_web_search(query)

    # OFFLINE
    offline_statute = offline_statute_lookup(query, selected_date)
    offline_context = build_offline_context(offline_statute)

    # WEB
    if use_web:
        web_results = serpapi_search(query)
        web_context = build_web_context(web_results)
    else:
        web_context = "Web search not required for this query."

    # ANSWER
    answer = generate_answer(query, web_context, offline_context)

    # DISPLAY
    st.markdown("### ✅ Final Answer")
    st.write(answer)

    st.markdown("### 📄 Offline Source")
    st.code(offline_context)

    st.markdown("### 🌐 Web Sources")
    st.code(web_context)

# --- 2. IMPROVED FUNCTIONS ---
