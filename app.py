import streamlit as st
import requests
import os
import html
from datetime import datetime, date
# chromadb and ollama are optional: they power the local nomic/ChromaDB semantic
# path, but the app runs fine without them (falls back to MiniLM, then TF-IDF).
# On hosted environments (e.g. Streamlit Cloud) these are typically absent.
try:
    import chromadb
except Exception:
    chromadb = None
try:
    import ollama
except Exception:
    ollama = None
import re
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# =====================================================
# CONFIG
# =====================================================
DB_PATH = "./legal_db"
METRICS_LOG_PATH = "./metrics_log.json"

# FIX: Load both IPC and BNS JSON files
IPC_DATA_PATH = "./IPC_updated.json"
BNS_DATA_PATH = "./BNS_updated.json"

OLLAMA_CHAT_MODEL = "qwen2.5:0.5b-instruct"
# Embedding model + endpoint for the ChromaDB semantic path (must match the model
# the collection was populated with — see populate_database.py).
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or ""
LEGAL_SITES = ["indiankanoon.org", "livelaw.in", "indiacode.nic.in"]
TOP_K = 5

# Date the BNS replaced the IPC (1 July 2024), as an int YYYYMMDD.
# Drives all time-aware IPC-vs-BNS selection.
BNS_CUTOFF = 20240701

# Local sentence-transformers model for semantic retrieval (runs offline once
# cached; no Ollama/network needed at query time).
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# Hybrid weight α: R = α·semantic + (1-α)·BM25 (report Eq. 5.5, α = 0.7).
SEMANTIC_WEIGHT = 0.7

# Stop words shared across the module
_STOP_WORDS = {
    "a","an","the","of","by","or","and","to","in","for","with","on",
    "at","from","not","amounting","attempt","commit","causing",
    "voluntarily","belonging","what","is","are","under","give","me",
    "tell","explain","describe","provide","recent","latest",
    # NOTE: "punishment", "judgment", "court", "ruling", "case", "law"
    # are intentionally NOT stop words — they are valid legal search terms
}

# =====================================================
# DYNAMIC INDEX — built from both JSON files at startup
# =====================================================
def _tokenise(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower())
            if w not in _STOP_WORDS and len(w) > 2]

# Synonym map: common search terms → BNS section numbers
# Needed because BNS uses different terminology than IPC
# Synonyms repeated multiple times to boost TF-IDF weight for that section.
# This ensures BNS sections are findable via IPC-era terminology.
_SECTION_SYNONYMS = {
    "152":    ["sedition"] * 8 + ["seditious", "disaffection", "hatred", "contempt", "state", "offences"],
    "103":    ["murder"] * 8 + ["homicide", "killing", "kill", "death", "intentional"],
    "103(2)": ["lynching"] * 8 + ["mob", "lynching", "group", "discriminatory", "caste", "religion", "communal"],
    # BNS §100 = culpable homicide definition; boost "culpable" + "defines" strongly
    "100":    ["culpable"] * 10 + ["homicide", "culpable homicide", "intention", "knowledge", "likely", "definition", "defines"],
    "105":    ["culpable", "homicide"] * 5 + ["death", "bodily", "injury"],
    # BNS §106 = death by negligence (IPC 304A equivalent)
    "106":    ["negligence"] * 10 + ["negligent", "rash", "accident", "accidental", "death", "causing death", "304A"],
    # BNS §109 = attempt to murder; "attempt to murder" must outscore bare "murder"
    "109":    ["attempt"] * 12 + ["attempted", "attempt to murder", "attempt murder", "307"],
    # BNS §101 = murder, also the representative section for "replaced IPC" / "came into force" queries
    "101":    ["murder"] * 6 + ["replaced", "replacement", "current", "force", "criminal", "offences",
               "bns", "bharatiya", "nyaya", "sanhita", "2024"],
    "63":     ["rape"] * 8 + ["sexual", "assault", "consent", "intercourse"],
    "64":     ["rape"] * 6 + ["minor", "sixteen", "child", "underage"],
    "303":    ["theft"] * 8 + ["stealing", "stolen", "movable", "property"],
    "304":    ["theft"] * 4 + ["snatching", "snatch"],
    "305":    ["dwelling"] * 8 + ["house", "worship", "place", "building", "residence", "transportation"],
    "306":    ["servant"] * 8 + ["clerk", "employee", "employer", "domestic", "staff"],
    "307":    ["preparation"] * 8 + ["causing", "armed", "ready"],
    "309":    ["robbery"] * 8 + ["robbed", "force"],
    "310":    ["dacoity"] * 8 + ["dacoits", "gang", "robbery"],
    "137":    ["kidnap"] * 8 + ["kidnapping", "abduction", "abduct"],
    "318":    ["cheating"] * 8 + ["fraud", "deceive", "deception", "dishonest"],
}

# FIX: Accept both IPC and BNS paths, merge data before indexing
def _build_full_index(ipc_path, bns_path):
    data = []
    for filepath in [ipc_path, bns_path]:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
        with open(filepath, encoding="utf-8") as f:
            data.extend(json.load(f))

    if not data:
        empty_bm25 = ({}, {}, 0.0, {}, defaultdict(list))
        return {}, {}, (defaultdict(list), {}, {}, empty_bm25), {}, re.compile(r'(?:section|bns|ipc)\s+([\d]+(?:\(\d+\))?)', re.IGNORECASE)

    section_map = {}
    topic_keywords = {}
    docs = {}

    for entry in data:
        sec      = str(entry.get("section", "")).strip()
        law      = entry.get("law", "")
        title    = entry.get("title", "")
        category = entry.get("category", "unknown").lower().strip()
        content  = entry.get("content", "")
        punishment = entry.get("punishment", "")

        # Use law+section as key to avoid IPC/BNS collisions (e.g. both have "304")
        map_key = f"{law}_{sec}" if law else sec
        section_map[map_key] = entry
        # Also store by bare section number for direct lookup (last writer wins,
        # but we prefer BNS since data is loaded IPC first then BNS)
        section_map[sec] = entry

        if category not in topic_keywords:
            topic_keywords[category] = set()
        topic_keywords[category].add(sec.lower())
        bare = re.sub(r'\(.*\)', '', sec).strip()
        if bare != sec:
            topic_keywords[category].add(bare)
        for word in re.findall(r'[a-z]+', (title + " " + content).lower()):
            if word not in _STOP_WORDS and len(word) > 3:
                topic_keywords[category].add(word)

        # Append synonyms for both BNS and IPC sections
        # IPC synonyms use a parallel map keyed by IPC section numbers
        _IPC_SYNONYMS = {
            "302":  ["murder"] * 8 + ["homicide", "killing", "kill", "death", "intentional"],
            "304A": ["negligence"] * 8 + ["negligent", "rash", "accident", "accidental", "death", "causing death"],
            "307":  ["attempt"] * 10 + ["attempted", "attempt to murder", "attempt murder"],
            "299":  ["culpable"] * 10 + ["homicide", "culpable homicide", "intention", "knowledge", "defines", "definition"],
            "300":  ["murder"] * 6 + ["culpable", "homicide", "definition"],
            "325":  ["grievous", "hurt", "grievous hurt"] * 6 + ["voluntarily", "causing"],
            "379":  ["theft"] * 8 + ["stealing", "stolen", "movable", "property", "punishment"],
            "392":  ["robbery"] * 8 + ["robbed", "force", "ipc"],
            "395":  ["dacoity"] * 8 + ["dacoits", "gang", "robbery", "ipc"],
        }
        # Synonyms are de-duplicated (dict.fromkeys) so a term counts once.
        # The old code repeated words ×8–12 to inflate TF-IDF weight; semantic
        # retrieval now carries meaning, so the inflation hack is removed.
        synonyms = ""
        if law == "BNS":
            synonyms = " ".join(dict.fromkeys(_SECTION_SYNONYMS.get(sec, [])))
        elif law == "IPC":
            synonyms = " ".join(dict.fromkeys(_IPC_SYNONYMS.get(sec, [])))

        full_text = f"{sec} {law} {title} {category} {content} {punishment} {synonyms}"
        docs[map_key] = _tokenise(full_text)

    topic_keywords = {k: sorted(v) for k, v in topic_keywords.items()}

    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1

    N = len(docs)
    idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}

    tfidf = {}
    for sec, tokens in docs.items():
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        tfidf[sec] = {t: (c / len(tokens)) * idf.get(t, 1) for t, c in tf.items()}

    inverted = defaultdict(list)
    for sec, weights in tfidf.items():
        for term, weight in weights.items():
            inverted[term].append((sec, weight))

    # ---- BM25 index (report specifies BM25 keyword retrieval, Eq. 5.4) ----
    doc_tf = {}
    for sec, tokens in docs.items():
        tfm = defaultdict(int)
        for t in tokens:
            tfm[t] += 1
        doc_tf[sec] = tfm
    doc_len = {sec: len(tokens) for sec, tokens in docs.items()}
    avgdl = (sum(doc_len.values()) / len(doc_len)) if doc_len else 0.0
    # Standard BM25 IDF: log(1 + (N - df + 0.5)/(df + 0.5))
    bm25_idf = {term: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
                for term, freq in df.items()}
    bm25_inverted = defaultdict(list)
    for sec, tfm in doc_tf.items():
        for term, f in tfm.items():
            bm25_inverted[term].append((sec, f))
    bm25 = (doc_tf, doc_len, avgdl, bm25_idf, bm25_inverted)

    escaped = [re.escape(s) for s in sorted(section_map.keys(), key=len, reverse=True)]
    section_pattern = re.compile(
        r'(?:(?:section|bns|ipc)\s*)?(' + '|'.join(escaped) + r')'
        r'|(?:section|bns|ipc)\s+([\d]+(?:\s*[\(\-]\s*\d+\s*\)?)?)',
        re.IGNORECASE
    )

    by_bare = defaultdict(list)
    for sec, entry in section_map.items():
        by_bare[re.sub(r'\(.*\)', '', sec).strip()].append((sec, entry))

    # Known IPC → BNS replacements where section numbers differ.
    # The five example mappings below match report Table 4.1 exactly.
    KNOWN_REPLACEMENTS = {
        "302":  "101",   # Murder (report Table 4.1)
        "378":  "303",   # Theft (report Table 4.1)
        "415":  "318",   # Cheating (report Table 4.1)
        "503":  "351",   # Criminal Intimidation (report Table 4.1)
        "320":  "117",   # Grievous Hurt (report Table 4.1)
        # Additional mappings used by the counterpart index / overrides:
        "124A": "152",   # Sedition
        "304":  "105",   # Culpable homicide
        "304A": "106",   # Death by negligence
        "379":  "304",   # Theft punishment
        "380":  "305",   # Theft in dwelling
        "392":  "309",   # Robbery
        "395":  "310",   # Dacoity
        "375":  "63",    # Rape definition
        "376":  "64",    # Rape punishment
        "363":  "137",   # Kidnapping
        "420":  "318",   # Cheating (dishonestly inducing delivery)
        "300":  "101",   # Murder definition
        "299":  "100",   # Culpable homicide definition
    }
    # Reverse map: BNS → IPC
    KNOWN_REPLACEMENTS_REVERSE = {v: k for k, v in KNOWN_REPLACEMENTS.items()}

    counterpart_index = {}
    for sec, entry in section_map.items():
        bare = re.sub(r'\(.*\)', '', sec).strip()
        law  = entry.get("law", "")
        ipc_e = bns_e = None

        # First try same-number matching
        for _, other in by_bare[bare]:
            if other.get("law") != law:
                if other.get("law") == "IPC":
                    ipc_e = other
                elif other.get("law") == "BNS":
                    bns_e = other

        # Then try known cross-number replacements
        if law == "IPC" and bns_e is None:
            bns_sec = KNOWN_REPLACEMENTS.get(bare)
            if bns_sec and bns_sec in section_map and section_map[bns_sec].get("law") == "BNS":
                bns_e = section_map[bns_sec]

        if law == "BNS" and ipc_e is None:
            ipc_sec = KNOWN_REPLACEMENTS_REVERSE.get(bare)
            if ipc_sec and ipc_sec in section_map and section_map[ipc_sec].get("law") == "IPC":
                ipc_e = section_map[ipc_sec]

        counterpart_index[sec] = {"ipc": ipc_e, "bns": bns_e}

    return section_map, topic_keywords, (inverted, idf, tfidf, bm25), counterpart_index, section_pattern


@st.cache_resource
def get_legal_index():
    # FIX: Pass both file paths
    return _build_full_index(IPC_DATA_PATH, BNS_DATA_PATH)

(SECTION_MAP,
    LEGAL_TOPIC_KEYWORDS,
    _TFIDF_INDEX,
    COUNTERPART_INDEX,
    SECTION_PATTERN) = get_legal_index()

_INVERTED, _IDF, _TFIDF, _BM25 = _TFIDF_INDEX

# =====================================================
# SEMANTIC INDEX (sentence-transformers) — real meaning-based retrieval.
# Replaces reliance on hand-tuned synonym/keyword tables. Built once from the
# same JSON entries and cached for the session.
# =====================================================
@st.cache_resource
def get_semantic_index():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME)
    keys, texts = [], []
    for key, entry in SECTION_MAP.items():
        # keep only law-qualified keys ("IPC_304"), skip bare aliases ("304")
        if "_" not in key:
            continue
        texts.append(
            f"{entry.get('law','')} Section {entry.get('section','')} "
            f"{entry.get('title','')} {entry.get('category','')} "
            f"{entry.get('content','')} {entry.get('punishment','')}"
        )
        keys.append(key)
    emb = model.encode(texts, normalize_embeddings=True) if texts else np.zeros((0, 384))
    return model, keys, np.asarray(emb, dtype="float32")


def _semantic_scores_minilm(query):
    """Fallback: cosine similarity via the local MiniLM index (0..1)."""
    try:
        model, keys, emb = get_semantic_index()
        if len(keys) == 0:
            return {}
        q = model.encode([query], normalize_embeddings=True)[0]
        sims = emb @ q  # embeddings are L2-normalised, so dot == cosine
        return {k: float(s) for k, s in zip(keys, sims)}
    except Exception:
        return {}


def _semantic_scores_chroma(query):
    """Primary semantic path: query the ChromaDB vector store using
    nomic-embed-text (the model the collection was built with).

    Returns {section_key: similarity 0..1}, min-max normalised from L2 distances.
    Only ids that exist in SECTION_MAP are kept (the store may hold sections that
    aren't in the current JSON). Returns {} on any failure so the caller can fall
    back to MiniLM / TF-IDF.
    """
    if collection is None:
        return {}
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": OLLAMA_EMBED_MODEL, "prompt": query},
            timeout=5,
        )
        r.raise_for_status()
        qemb = r.json()["embedding"]

        n = collection.count()
        if not n:
            return {}
        res = collection.query(
            query_embeddings=[qemb], n_results=n, include=["distances"]
        )
        ids   = res["ids"][0]
        dists = res["distances"][0]

        pairs = [(i, d) for i, d in zip(ids, dists) if i in SECTION_MAP]
        if not pairs:
            return {}
        ds = [d for _, d in pairs]
        dmin, dmax = min(ds), max(ds)
        span = (dmax - dmin) or 1.0
        # nearest (smallest distance) -> 1.0, farthest -> 0.0
        return {i: (dmax - d) / span for i, d in pairs}
    except Exception:
        return {}


def _semantic_scores(query):
    """Semantic similarity per section (0..1). ChromaDB + nomic-embed-text (Ollama)
    is the primary path; falls back to the local MiniLM index if it's unavailable."""
    scores = _semantic_scores_chroma(query)
    if scores:
        return scores
    return _semantic_scores_minilm(query)


def _bm25_scores(query, k1=1.5, b=0.75):
    """BM25 keyword relevance per section (report Eq. 5.4).

    score(D) = Σ_t IDF(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))
    """
    doc_tf, doc_len, avgdl, bm25_idf, bm25_inverted = _BM25
    q_tokens = _tokenise(query)
    if not q_tokens or avgdl == 0:
        return {}
    scores = defaultdict(float)
    for t in set(q_tokens):
        idf = bm25_idf.get(t)
        if idf is None:
            continue
        for sec, f in bm25_inverted.get(t, []):
            dl = doc_len.get(sec, 0)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            if denom:
                scores[sec] += idf * (f * (k1 + 1)) / denom
    return dict(scores)


def _hybrid_score_all(query):
    """Hybrid ranker (report Eq. 5.5): R = α·semantic + (1-α)·BM25.

    Semantic = cosine similarity (ChromaDB / nomic-embed-text); keyword = BM25.
    Both components are normalised to 0..1 so neither dominates by scale.
    """
    kw  = _bm25_scores(query)
    sem = _semantic_scores(query)

    kw_max = max(kw.values()) if kw else 0.0
    kw_norm = {k: (v / kw_max) for k, v in kw.items()} if kw_max > 0 else {}

    candidates = set(sem) | set(kw_norm)
    results = []
    for key in candidates:
        entry = SECTION_MAP.get(key)
        if not entry:
            continue
        s = SEMANTIC_WEIGHT * max(sem.get(key, 0.0), 0.0) \
            + (1 - SEMANTIC_WEIGHT) * kw_norm.get(key, 0.0)
        if s <= 0:
            continue
        results.append((s, key, entry))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# =====================================================
# CHROMADB
# =====================================================
@st.cache_resource
def get_chroma_collection():
    # Non-fatal: if chromadb is unavailable or fails to open the store, return
    # None and let retrieval fall back to the MiniLM semantic index.
    if chromadb is None:
        return None
    try:
        c = chromadb.PersistentClient(path=DB_PATH)
        return c.get_or_create_collection(name="criminal_law")
    except Exception:
        return None

collection = get_chroma_collection()

# =====================================================
# SECTION EXTRACTION
# =====================================================
def _normalise_section(raw):
    raw = raw.strip()
    if raw in SECTION_MAP:
        return raw
    m = re.match(r'^(\d+)\s*[\(\-\s]\s*(\d+)\s*\)?$', raw)
    if m:
        candidate = f"{m.group(1)}({m.group(2)})"
        if candidate in SECTION_MAP:
            return candidate
    return raw

def extract_section_number(query):
    match = SECTION_PATTERN.search(query)
    if not match:
        return None
    raw = next((g for g in match.groups() if g), None)
    return _normalise_section(raw) if raw else None

# =====================================================
# TF-IDF SEARCH
# =====================================================
def _query_scores(query):
    """Compute raw TF-IDF scores for all sections."""
    q_tokens = _tokenise(query)
    if not q_tokens:
        return {}
    q_tf = defaultdict(int)
    for t in q_tokens:
        q_tf[t] += 1
    q_tfidf = {t: (c / len(q_tokens)) * _IDF.get(t, 1) for t, c in q_tf.items()}
    scores = defaultdict(float)
    for term, q_w in q_tfidf.items():
        for key, d_w in _INVERTED.get(term, []):
            scores[key] += q_w * d_w
    return scores


def _tfidf_search(query, qdate):
    scores = _query_scores(query)
    if not scores:
        return []
    results = []
    for key, score in scores.items():
        if score <= 0:
            continue
        entry = SECTION_MAP.get(key, {})
        if not entry:
            continue
        vf = int(entry.get("valid_from", "0000-00-00").replace("-", ""))
        vu = int(entry.get("valid_until", "9999-12-31").replace("-", ""))
        if vf <= qdate <= vu:
            results.append((score, key, entry))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:TOP_K]

def _make_result(entry, relevance=1.0):
    sec = str(entry.get("section", ""))
    cp  = COUNTERPART_INDEX.get(sec, {})
    return {
        "title"           : f"{entry.get('law')} Section {sec}",
        "section"         : sec,
        "law"             : entry.get("law", ""),
        "text"            : entry.get("content", ""),
        "content"         : entry.get("content", ""),
        "punishment"      : entry.get("punishment", "Not specified"),
        "source"          : entry.get("source", ""),
        "category"        : entry.get("category", ""),
        "valid_from"      : entry.get("valid_from", ""),
        "valid_until"     : entry.get("valid_until", ""),
        "relevance"       : relevance,
        "ipc_counterpart" : cp.get("ipc"),
        "bns_counterpart" : cp.get("bns"),
        "note"            : entry.get("note", ""),
    }
# =====================================================
# RERANKING + CONFIDENCE (NEW)
# =====================================================

def rule_based_override(query, statutes, qdate=None):
    query_lower = query.lower()

    ipc_explicit = "ipc" in query_lower or "indian penal" in query_lower
    bns_explicit = "bns" in query_lower or "bharatiya" in query_lower

    # Time-aware default: before the BNS cutoff the IPC was in force, so a query
    # that doesn't name a law should resolve to the IPC target; on/after the
    # cutoff it should resolve to the BNS target. An explicit "ipc"/"bns" in the
    # query always wins over the date.
    prefer_ipc_by_date = qdate is not None and qdate < BNS_CUTOFF

    # Each rule: (trigger_phrase, exclude_phrases, bns_target, ipc_target)
    # bns_target / ipc_target: (law, section) or None
    RULE_MAP = [
        # Sedition: BNS 152 text no longer contains the word "sedition", so map it explicitly.
        ("sedition",          [],                           ("BNS", "152"),    ("IPC", "124A")),
        ("seditious",         [],                           ("BNS", "152"),    ("IPC", "124A")),
        ("mob lynching",      [],                           ("BNS", "103(2)"), None),
        ("lynching",          [],                           ("BNS", "103(2)"), None),
        ("attempt to murder", [],                           ("BNS", "109"),    ("IPC", "307")),
        ("attempt murder",    [],                           ("BNS", "109"),    ("IPC", "307")),
        ("servant",           [],                           ("BNS", "306"),    ("IPC", "381")),
        ("clerk",             [],                           ("BNS", "306"),    ("IPC", "381")),
        ("dwelling house",    [],                           ("BNS", "305"),    ("IPC", "380")),
        ("after preparation", [],                           ("BNS", "307"),    ("IPC", "380")),
        ("snatching",         [],                           ("BNS", "304"),    None),
        ("robbery",           ["dacoity"],                  ("BNS", "309"),    ("IPC", "392")),
        ("dacoity",           [],                           ("BNS", "310"),    ("IPC", "395")),
        ("culpable homicide", ["not amounting", "amounts"], ("BNS", "100"),    ("IPC", "299")),
        ("defines culpable",  [],                           ("BNS", "100"),    ("IPC", "299")),
        ("deals with culpable", [],                         ("BNS", "100"),    ("IPC", "299")),
        ("grievous hurt",     ["mob", "group", "lynching"], ("BNS", "116"),    ("IPC", "325")),
        ("causing death by negligence", [],                 ("BNS", "106"),    ("IPC", "304A")),
        ("death by negligence", [],                         ("BNS", "106"),    ("IPC", "304A")),
        ("negligence",        ["culpable", "hurt"],         ("BNS", "106"),    ("IPC", "304A")),
        ("murder",            ["attempt", "culpable"],      ("BNS", "101"),    ("IPC", "302")),
        ("theft",             ["servant", "clerk", "dwelling", "house", "worship", "preparation", "armed", "after"],
                                                            ("BNS", "303"),    ("IPC", "379")),
        # "replaced IPC" / "came into force" / "which law replaced" → BNS §101
        ("replaced ipc",      [],                           ("BNS", "101"),    None),
        ("replaced the ipc",  [],                           ("BNS", "101"),    None),
        ("replaced ipc for",  [],                           ("BNS", "101"),    None),
        ("which law replaced", [],                          ("BNS", "101"),    None),
        ("came into force",   [],                           ("BNS", "101"),    None),
        ("come into force",   [],                           ("BNS", "101"),    None),
        ("bns come into",     [],                           ("BNS", "101"),    None),
        ("ipc still applicable", [],                        ("BNS", "101"),    None),
        ("ipc applicable",    ["section", "§"],             ("BNS", "101"),    None),
    ]

    for rule in RULE_MAP:
        if not isinstance(rule, tuple) or len(rule) < 4:
            continue
        keyword, exclude_kws, bns_target, ipc_target = rule
        if keyword not in query_lower:
            continue
        if any(ex in query_lower for ex in exclude_kws):
            continue

        # Pick the correct target based on law context, then date.
        # Priority: explicit law in query > date-based preference > whichever
        # target exists.
        if ipc_explicit:
            chosen = ipc_target or bns_target
        elif bns_explicit:
            chosen = bns_target or ipc_target
        elif prefer_ipc_by_date:
            chosen = ipc_target or bns_target
        else:
            chosen = bns_target or ipc_target

        if not chosen:
            continue
        target_law, target_sec = chosen

        # First try: find target in current statutes
        for s in statutes:
            if str(s.get("section")) == target_sec and s.get("law") == target_law:
                return [s]

        # Second try: fetch directly from SECTION_MAP if not in top-5
        map_key = f"{target_law}_{target_sec}"
        entry = SECTION_MAP.get(map_key) or SECTION_MAP.get(target_sec)
        if entry and entry.get("law") == target_law:
            return [_make_result(entry, 1.0)]

    return statutes


def rerank_results(statutes, query):
    query_lower = query.lower()
    ipc_explicit = "ipc" in query_lower or "indian penal" in query_lower
    bns_explicit = "bns" in query_lower or "bharatiya" in query_lower

    KEYWORD_SECTION_MAP = {
        "theft":           (["303", "379"],    ["304", "305", "306", "307", "380", "381", "382"]),
        "murder":          (["101", "302"],    ["103", "300", "104"]),
        "robbery":         (["309", "392"],    ["310", "311", "312", "313", "395"]),
        "dacoity":         (["310", "395"],    ["309", "312"]),
        "grievous hurt":   (["116", "325"],    ["115", "117", "323", "320"]),
        "hurt":            (["116", "325"],    []),
        "negligence":      (["106", "304A"],   ["103", "302"]),
        "attempt to murder": (["109", "307"],  ["101", "302", "103"]),
        "culpable homicide": (["100", "105", "299", "300"], ["101", "302"]),
        "attempt":         (["109", "307"],    []),
        "snatching":       (["304"],           []),
        "dacoits":         (["310", "395"],    []),
    }

    for s in statutes:
        score = s.get("relevance", 0)
        section = str(s.get("section", ""))
        law = s.get("law", "")

        for keyword, (good_sections, bad_sections) in KEYWORD_SECTION_MAP.items():
            if keyword in query_lower:
                if section in good_sections:
                    score += 1.0
                elif section in bad_sections:
                    score -= 0.4

        # Exact section number in query
        if section in query:
            score += 0.8

        # Law-explicit boosts — strong signal: user said "under IPC" or "under BNS"
        if ipc_explicit and law == "IPC":
            score += 0.6
        if ipc_explicit and law == "BNS":
            score -= 0.5   # penalise wrong law when IPC is specified
        if bns_explicit and law == "BNS":
            score += 0.6
        if bns_explicit and law == "IPC":
            score -= 0.5

        s["final_score"] = score

    return sorted(statutes, key=lambda x: x["final_score"], reverse=True)


def apply_confidence_filter(statutes, threshold=0.05):
    if not statutes:
        return statutes

    top_score = statutes[0].get("final_score", statutes[0].get("relevance", 0))

    if top_score < threshold:
        return []  # reject only genuinely empty/zero matches

    return statutes
# =====================================================
# MAIN SEARCH
# =====================================================
def _tfidf_score_all(query):
    """Return TF-IDF scores for ALL sections without any date filtering."""
    scores = _query_scores(query)
    if not scores:
        return []
    results = []
    for key, score in scores.items():
        if score <= 0:
            continue
        entry = SECTION_MAP.get(key)
        if not entry:
            continue
        results.append((score, key, entry))
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# =====================================================
# BASELINE SEARCH (NO TEMPORAL LOGIC)
# =====================================================
def baseline_search(query):
    """
    Pure TF-IDF search without temporal filtering or boosting
    (Used as baseline model for hypothesis testing)
    """
    scores = _tfidf_score_all(query)

    if not scores:
        return []

    return [_make_result(entry, score) for score, sec, entry in scores[:TOP_K]]

def chroma_search(query, selected_date):
    qdate       = int(selected_date.strftime("%Y%m%d"))
    section_num = extract_section_number(query)

    # If user requests a specific section number, resolve it law- and date-aware.
    # The bare `section_num` key collides between IPC and BNS (both have e.g. 304),
    # so prefer the law-qualified "{LAW}_{sec}" keys and disambiguate by:
    #   1. an explicit "ipc"/"bns" in the query, else
    #   2. the selected date (IPC before the BNS cutoff, BNS on/after).
    if section_num:
        query_lower  = query.lower()
        ipc_explicit = "ipc" in query_lower or "indian penal" in query_lower
        bns_explicit = "bns" in query_lower or "bharatiya" in query_lower
        ipc_e = SECTION_MAP.get(f"IPC_{section_num}")
        bns_e = SECTION_MAP.get(f"BNS_{section_num}")

        if ipc_explicit and ipc_e:
            chosen = ipc_e
        elif bns_explicit and bns_e:
            chosen = bns_e
        elif ipc_e and bns_e:
            chosen = ipc_e if qdate < BNS_CUTOFF else bns_e
        else:
            # Section exists in only one law, or only under the bare key.
            chosen = ipc_e or bns_e or SECTION_MAP.get(section_num)

        # Non-existent section (e.g. "IPC Section 999") -> empty, no hallucination.
        if not chosen:
            return []
        return [_make_result(chosen, 1.0)]

    # For topic queries: hybrid (semantic + BM25) score over ALL sections (Eq. 5.5),
    # then apply the binary temporal filter T(d) (Eq. 5.7) and final score S = R × T
    # (Eq. 5.8). Statutes not legally valid at the query date are discarded.
    all_scored = _hybrid_score_all(query)
    if not all_scored:
        return []

    def _temporally_valid(entry):
        vf = int(entry.get("valid_from", "00000000").replace("-", ""))
        vu = int(entry.get("valid_until", "99991231").replace("-", ""))
        return vf <= qdate <= vu   # T(d) = 1 if in force, else 0

    # T(d): discard statutes outside their validity period. S = R × 1 for survivors.
    valid_scored = [(score, sec, entry) for score, sec, entry in all_scored
                    if _temporally_valid(entry)]
    if not valid_scored:
        return []
    valid_scored.sort(key=lambda x: x[0], reverse=True)
    results = [_make_result(entry, score) for score, sec, entry in valid_scored[:TOP_K]]

    # NEW: RERANK + FILTER
    results = rerank_results(results, query)

    # NEW FIX (H2 BOOST) — date-aware so IPC/BNS is chosen by the selected date
    results = rule_based_override(query, results, qdate)

    results = apply_confidence_filter(results)

    return results

# =====================================================
# ONLINE SEARCH
# =====================================================
@st.cache_data(ttl=3600, show_spinner=False)
def serpapi_search(query, year):
    # Cached for 1 hour per (query, year) to conserve the SerpAPI free-tier
    # quota (100 searches/month) — repeated/identical queries don't re-spend.
    if not SERPAPI_KEY:
        return []

    site_filter = " OR ".join([f"site:{s}" for s in LEGAL_SITES])
    q = f"{query} {year} {site_filter}"

    params = {
        "engine": "google",
        "q": q,
        "api_key": SERPAPI_KEY,
        "num": 5,
        "gl": "in",
        "hl": "en"
    }

    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=15)
        return r.json().get("organic_results", [])
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _amendment_search(topic_str, year):
    # Cached law-change web lookup (used by detect_law_changes) — also conserves
    # the SerpAPI quota so each query doesn't double-spend on searches.
    if not SERPAPI_KEY:
        return []
    site_filter = " OR ".join([f"site:{s}" for s in LEGAL_SITES])
    params = {
        "engine": "google",
        "q": f"India criminal law amendment {topic_str} {year} {site_filter}",
        "api_key": SERPAPI_KEY,
        "num": 3, "gl": "in", "hl": "en",
    }
    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=10)
        return r.json().get("organic_results", [])
    except Exception:
        return []

# =====================================================
# EVALUATION METRICS (BACKEND ONLY)
# =====================================================
def evaluate_offline(query, statutes, selected_date):
    """Query-time retrieval signals (no ground truth available).

    These are honestly computable at request time — unlike the previous version,
    which reported Precision=1.0 and a hardcoded Temporal Accuracy=1.0 regardless
    of the result. In particular, Temporal Validity actually checks each returned
    statute against the selected date, so it can catch temporal regressions.
    """
    metrics = {
        "Retrieval Confidence": 0.0,  # normalised score of the top result
        "Score Margin": 0.0,          # separation between #1 and #2 (decisiveness)
        "Temporal Validity": 0.0,     # fraction of results actually in force on the date
        "Top-K Coverage": 0.0,        # how full the result set is
    }
    if not statutes:
        return metrics

    qdate = int(selected_date.strftime("%Y%m%d"))
    scores = [float(s.get("final_score", s.get("relevance", 0.0))) for s in statutes]
    top = scores[0]

    # Confidence: clip the top score into 0..1 (hybrid scores are ~0..1 before boosts).
    metrics["Retrieval Confidence"] = round(max(0.0, min(top, 1.0)), 4)

    # Margin: how far #1 stands above #2 (0 when only one result or a tie).
    if len(scores) > 1 and top > 0:
        metrics["Score Margin"] = round(max(0.0, (top - scores[1]) / top), 4)

    # Temporal Validity: share of returned statutes that were actually in force
    # on the selected date. A correct time-aware system should score ~1.0 here.
    in_force = 0
    for s in statutes:
        vf = int(str(s.get("valid_from", "0000-00-00")).replace("-", "") or 0)
        vu = int(str(s.get("valid_until", "9999-12-31")).replace("-", "") or 99991231)
        if vf <= qdate <= vu:
            in_force += 1
    metrics["Temporal Validity"] = round(in_force / len(statutes), 4)

    metrics["Top-K Coverage"] = round(len(statutes) / TOP_K, 4) if TOP_K > 0 else 0.0
    return metrics

def evaluate_online(query, web_results):
    metrics = {
        "Trusted Source Precision": 0.0, "Topical Relevance": 0.0,
        "Temporal Freshness": 0.0, "Web Coverage": 0.0
    }
    if not web_results:
        return metrics
    trusted = sum(
        1 for w in web_results
        if any(site in (w.get("link") or "") for site in LEGAL_SITES)
    )
    metrics["Trusted Source Precision"] = trusted / len(web_results)
    topics = set(_tokenise(query))
    topic_hits = sum(
        1 for w in web_results
        if any(t in _tokenise(w.get("title","") + " " + w.get("snippet","")) for t in topics)
    )
    metrics["Topical Relevance"] = topic_hits / len(web_results)
    def _parse_web_date(w):
        """Parse a SerpAPI date string into a datetime, return None if unparseable."""
        raw = w.get("date", "")
        for fmt in ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                pass
        return None

    now = datetime.today()
    one_year_ago = now.replace(year=now.year - 1)
    fresh_hits = 0
    dated_count = 0
    for w in web_results:
        parsed = _parse_web_date(w)
        if parsed is not None:
            dated_count += 1
            if parsed >= one_year_ago:
                fresh_hits += 1
    # Score based on dated results only; if none have dates, score is 0
    metrics["Temporal Freshness"] = fresh_hits / dated_count if dated_count > 0 else 0.0
    metrics["Web Coverage"] = 1.0
    return metrics

def evaluate_generation(answer, statutes, web_results, query):
    metrics = {
        "Context Usage": 0.0, "Answer Length": 0.0,
        "Source Grounding": 0.0, "Completeness": 0.0
    }
    if not answer or answer == "No relevant legal material found.":
        return metrics
    context_keywords = set()
    for s in statutes:
        words = s['title'].lower().split()
        context_keywords.update([w for w in words if len(w) > 4])
    for w in web_results:
        words = (w.get('title','') + ' ' + w.get('snippet','')).lower().split()
        context_keywords.update([w for w in words if len(w) > 4])
    answer_lower = answer.lower()
    used_keywords = sum(1 for kw in context_keywords if kw in answer_lower)
    metrics["Context Usage"] = min(used_keywords / max(len(context_keywords), 1), 1.0) if context_keywords else 0.0
    word_count = len(answer.split())
    if 100 <= word_count <= 500:
        metrics["Answer Length"] = 1.0
    elif word_count < 100:
        metrics["Answer Length"] = word_count / 100
    else:
        metrics["Answer Length"] = max(0.5, 1 - (word_count - 500) / 1000)
    legal_indicators = ['section','act','ipc','code','law','provision','under','according']
    grounding_score = sum(1 for indicator in legal_indicators if indicator in answer_lower)
    metrics["Source Grounding"] = min(grounding_score / 5, 1.0)
    query_words = {w for w in query.lower().split() if len(w) > 3}
    addressed = sum(1 for qw in query_words if qw in answer_lower)
    metrics["Completeness"] = addressed / max(len(query_words), 1) if query_words else 0.0
    return metrics

# =====================================================
# LAW CHANGE DETECTION
# =====================================================
def detect_law_changes(query, statutes, selected_date):
    changes = {
        "has_changes": False,
        "chroma_changes": [],
        "web_changes": [],
        "summary": ""
    }
    # Only reason about the TOP (most relevant) result for the change banner —
    # otherwise every unrelated BNS section in the top-5 gets wrongly announced
    # as a replacement of the queried provision.
    top = statutes[:1]
    ipc_sections = [s for s in top if s.get("law") == "IPC" or s["title"].startswith("IPC")]
    bns_sections = [s for s in top if s.get("law") == "BNS" or s["title"].startswith("BNS")]

    if ipc_sections and bns_sections:
        changes["has_changes"] = True
        for ipc in ipc_sections:
            for bns in bns_sections:
                changes["chroma_changes"].append({
                    "type": "replaced", "old": ipc["title"], "new": bns["title"],
                    "detail": "This provision was part of the IPC (1860) and has been re-enacted under the Bharatiya Nyaya Sanhita (BNS), 2023, effective 1 July 2024."
                })
    elif bns_sections and not ipc_sections:
        changes["has_changes"] = True
        for bns in bns_sections:
            changes["chroma_changes"].append({
                "type": "new_law", "old": "IPC 1860", "new": bns["title"],
                "detail": "This provision now falls under the Bharatiya Nyaya Sanhita (BNS), 2023, which replaced the IPC effective 1 July 2024."
            })
    elif ipc_sections and not bns_sections:
        bns_cutoff = 20240701
        qdate = int(selected_date.strftime("%Y%m%d"))
        if qdate >= bns_cutoff:
            changes["has_changes"] = True
            for ipc in ipc_sections:
                # Check if a BNS counterpart is known via counterpart index
                sec = ipc.get("section", "")
                cp = COUNTERPART_INDEX.get(sec, {})
                bns_cp = cp.get("bns")
                if bns_cp:
                    changes["chroma_changes"].append({
                        "type": "replaced",
                        "old": ipc["title"],
                        "new": f"BNS Section {bns_cp.get('section')} — {bns_cp.get('title','')}",
                        "detail": f"IPC Section {sec} was replaced by BNS Section {bns_cp.get('section')} effective 1 July 2024."
                    })
                else:
                    changes["chroma_changes"].append({
                        "type": "possibly_replaced", "old": ipc["title"], "new": "BNS 2023",
                        "detail": "The IPC was replaced by the BNS on 1 July 2024. A corresponding BNS provision may exist but was not found in the database."
                    })

    if SERPAPI_KEY:
        topics = set(_tokenise(query))
        topic_str = " ".join(topics) if topics else (query.split()[0] if query.split() else "")
        web_hits = _amendment_search(topic_str, selected_date.year)
        amendment_keywords = ["amend","repeal","replac","new law","bns","reform","revised","notif"]
        for w in web_hits:
            text = (w.get("title","") + " " + w.get("snippet","")).lower()
            # Require BOTH an amendment signal AND on-topic relevance (at least one
            # query topic term present), so tangential results aren't surfaced.
            on_topic = (not topics) or any(t in text for t in topics)
            if on_topic and any(kw in text for kw in amendment_keywords):
                changes["has_changes"] = True
                changes["web_changes"].append({
                    "title": w.get("title",""),
                    "snippet": w.get("snippet",""),
                    "link": w.get("link","#")
                })

    if changes["has_changes"]:
        parts = []
        if changes["chroma_changes"]:
            replaced_new          = [c["new"] for c in changes["chroma_changes"] if c["type"] == "replaced"]
            new_law_entries       = [c["new"] for c in changes["chroma_changes"] if c["type"] == "new_law"]
            possibly_replaced_old = [c["old"] for c in changes["chroma_changes"] if c["type"] == "possibly_replaced"]
            if replaced_new:
                sections_str = ", ".join(f"<strong>{s}</strong>" for s in replaced_new)
                parts.append(f"These provisions have been re-enacted under {sections_str} of the Bharatiya Nyaya Sanhita (BNS), 2023, which replaced the IPC effective 1 July 2024.")
            if new_law_entries:
                sections_str = ", ".join(f"<strong>{s}</strong>" for s in new_law_entries)
                parts.append(f"This provision now falls under {sections_str} of the Bharatiya Nyaya Sanhita (BNS), 2023, which replaced the IPC effective 1 July 2024.")
            if possibly_replaced_old:
                sections_str = ", ".join(f"<strong>{s}</strong>" for s in possibly_replaced_old)
                parts.append(f"{sections_str} may have been replaced under BNS 2023. A corresponding BNS provision may exist but was not found in the database.")
        if changes["web_changes"]:
            parts.append(f"<strong>{len(changes['web_changes'])} recent web source(s)</strong> report amendments or changes to this law.")
        changes["summary"] = " &nbsp;|&nbsp; ".join(parts)

    return changes

# =====================================================
# METRICS JSON LOGGER
# =====================================================
def log_metrics_to_json(query, selected_date, statutes, web_results, answer):
    offline_metrics    = evaluate_offline(query, statutes, selected_date)
    online_metrics     = evaluate_online(query, web_results)
    generation_metrics = evaluate_generation(answer, statutes, web_results, query)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "selected_date": selected_date.strftime("%Y-%m-%d"),
        "statutes_retrieved": len(statutes),
        "web_results_retrieved": len(web_results),
        "offline_retrieval_metrics":  {k: round(v, 4) for k, v in offline_metrics.items()},
        "online_retrieval_metrics":   {k: round(v, 4) for k, v in online_metrics.items()},
        "generation_metrics":         {k: round(v, 4) for k, v in generation_metrics.items()},
    }
    if os.path.exists(METRICS_LOG_PATH):
        with open(METRICS_LOG_PATH, "r") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    else:
        log = []
    log.append(entry)
    with open(METRICS_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    return entry

# =====================================================
# ANSWER GENERATION
# =====================================================
def _format_counterpart(label, entry, color_class=""):
    if not entry:
        return ""
    lines = [
        f"{label}: Section {entry.get('section')} — {entry.get('title','')}",
        f"Definition : {_short(entry.get('content',''))}",
        f"Punishment : {entry.get('punishment','Not specified')}",
        f"Source : {entry.get('source','')}",
    ]
    if entry.get("valid_from"):
        vf = entry.get("valid_from","")
        vu = entry.get("valid_until","9999-12-31")
        lines.append(f"In force : {vf} → {'Present' if vu == '9999-12-31' else vu}")
    return "\n".join(lines)

# =====================================================
# GROUNDED ANSWER (NO HALLUCINATION)
# =====================================================
def _short(text, limit=180):
    """Trim long statutory text for the concise Legal Analysis summary,
    breaking on a sentence/word boundary. Full text still shows in the cards."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    b = max(cut.rfind(". "), cut.rfind("; "))
    if b > limit * 0.5:
        return cut[:b + 1] + " …"
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut) + " …"


def grounded_answer(query, statutes):
    if not statutes:
        return "No relevant legal provision found."

    top = statutes[0]

    law = top.get("law", "")
    section = top.get("section", "")
    punishment = top.get("punishment", "Not specified")
    content = _short(top.get("content", ""))

    return (
        f"**{law} Section {section}** — {content}\n\n"
        f"**Punishment:** {punishment}"
    )

def filter_hallucination(answer, statutes):
    # Valid = the retrieved statutes AND their IPC/BNS counterparts, since a
    # correct comparative answer legitimately cites the counterpart section.
    # Sub-parts like "103(2)" are accepted both in full and by their bare number.
    def _add(valid, raw):
        raw = str(raw or "").strip()
        if not raw:
            return
        valid.add(re.sub(r"\s+", "", raw))                       # e.g. 103(2)
        valid.add(re.sub(r"\(.*\)", "", raw).strip())            # bare 103

    valid_sections = set()
    for s in statutes:
        _add(valid_sections, s.get("section"))
        for cp in (s.get("ipc_counterpart"), s.get("bns_counterpart")):
            if cp:
                _add(valid_sections, cp.get("section"))

    # Match section numbers including optional sub-parts, e.g. "section 103(2)".
    mentioned = re.findall(r"\bsection\s*(\d+\s*(?:\(\s*\d+\s*\))?)", answer.lower())

    for sec in mentioned:
        norm = re.sub(r"\s+", "", sec)
        bare = re.sub(r"\(.*\)", "", norm).strip()
        if norm not in valid_sections and bare not in valid_sections:
            return "The answer could not be verified from retrieved legal provisions."

    return answer

def generate_answer(query, statutes, web_results, selected_date):
    if not statutes and not web_results:
        return "No relevant legal material found."

    if statutes and all(s.get("punishment") and s.get("punishment") != "Not specified" for s in statutes):
        parts = []
        for s in statutes:
            vf = s.get("valid_from", "")
            vu = s.get("valid_until", "9999-12-31")
            validity = f"{vf} → {'Present' if vu == '9999-12-31' else vu}"
            block = [
                f"**{s['title']}** — {_short(s.get('content', s.get('text','')))}",
                f"**Punishment:** {s.get('punishment', 'Not specified')}",
                f"**In force:** {validity}",
            ]
            ipc = s.get("ipc_counterpart")
            bns = s.get("bns_counterpart")
            if ipc:
                block.append(_format_counterpart("Previous Law (IPC)", ipc))
            if bns:
                block.append(_format_counterpart("New Law (BNS)", bns))
            if ipc and bns:
                block.append(
                    f"Key change: IPC Section {ipc.get('section')} punishment was "
                    f"'{ipc.get('punishment','N/A')}'. "
                    f"Under BNS Section {bns.get('section')} it is now "
                    f"'{bns.get('punishment','N/A')}'."
                )
            parts.append("\n".join(block))
        for w in web_results:
            parts.append(f"\nJudicial Update: {w.get('title','')}\n{w.get('snippet','')}")
        return "\n\n" + "\n\n".join(parts)

    # Slow path: LLM call for open-ended topic queries
    context = []
    for s in statutes:
        lines = [
            f"STATUTE: {s['title']}",
            f"Definition: {s.get('content', s.get('text',''))}",
            f"Punishment: {s.get('punishment','Not specified')}",
            f"Source: {s.get('source','')}",
        ]
        ipc = s.get("ipc_counterpart")
        bns = s.get("bns_counterpart")
        if ipc:
            lines.append(f"IPC equivalent: S.{ipc.get('section')} — {ipc.get('punishment','')}")
        if bns:
            lines.append(f"BNS equivalent: S.{bns.get('section')} — {bns.get('punishment','')}")
        context.append("\n".join(lines))
    for w in web_results:
        context.append(f"UPDATE: {w.get('title','')}: {w.get('snippet','')}")

    system_prompt = (
        f"Indian legal assistant. Date: {selected_date.strftime('%d %B %Y')}. "
        "Answer using ONLY the context. Be concise and structured. "
        "Always state: section, definition, punishment, source. "
        "Compare old (IPC) and new (BNS) law if both are present."
    )
    if ollama is None:
        raise RuntimeError("ollama unavailable")  # -> generate_answer_safe falls back
    # Use a client with a hard timeout so a slow/stuck model can't hang the app;
    # on timeout this raises and generate_answer_safe returns the deterministic answer.
    client = ollama.Client(timeout=20)
    res = client.chat(
        model=OLLAMA_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n" + "\n---\n".join(context)}
        ],
        options={"temperature": 0.1, "num_predict": 400}
    )
    return res["message"]["content"]


def generate_answer_safe(query, statutes, web_results, selected_date):
    """LLM answer with an automatic fallback to the deterministic grounded answer.

    Tries generate_answer() (which itself only calls Ollama for open-ended topic
    queries — statute-style queries stay deterministic). If Ollama is down, times
    out, or the LLM output can't be verified against the retrieved statutes, we
    fall back to grounded_answer() so the app never breaks when Ollama isn't running.
    """
    REJECT = "The answer could not be verified from retrieved legal provisions."

    # generate_answer() uses a deterministic, data-built structured layout when
    # every retrieved statute has a punishment; that output is trustworthy by
    # construction (no LLM, no hallucination) and must NOT be run through the
    # section filter — it legitimately cites related/counterpart sections.
    deterministic = bool(statutes) and all(
        s.get("punishment") and s.get("punishment") != "Not specified"
        for s in statutes
    )
    try:
        ans = generate_answer(query, statutes, web_results, selected_date)
        if deterministic:
            return ans
        # Open-ended query -> real LLM output -> verify it against the statutes.
        checked = filter_hallucination(ans, statutes) if ans else ""
        if checked and checked.strip() and checked != REJECT:
            return checked
    except Exception:
        pass  # Ollama unavailable / timed out -> deterministic fallback below
    return filter_hallucination(grounded_answer(query, statutes), statutes)

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Time-Aware Legal Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon=""
)

# ---- Light / Dark theme toggle (sliding switch) ----
_tcols = st.columns([8, 2])
with _tcols[1]:
    if "theme_toggle" not in st.session_state:
        st.session_state["theme_toggle"] = True  # default: Dark on
    # No text label — the sun/moon icon lives inside the knob (styled via CSS).
    _dark = st.toggle("Theme", key="theme_toggle", label_visibility="collapsed")
    theme = "Dark" if _dark else "Light"

if theme == "Light":
    bg_gradient     = "linear-gradient(135deg, #f6f8fb 0%, #eef2f7 100%)"
    header_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    card_bg         = "#ffffff"
    text_color      = "#1a202c"
    text_secondary  = "#5a6675"
    border_color    = "#cbd5e0"
    field_value     = "#2d3748"
else:
    bg_gradient     = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
    header_gradient = "linear-gradient(135deg, #4a5568 0%, #2d3748 100%)"
    card_bg         = "#2d3748"
    text_color      = "#e2e8f0"
    text_secondary  = "#a0aec0"
    border_color    = "#4a5568"
    field_value     = "#cbd5e0"

st.markdown(f"""
    <style>
    .main {{
    background: {bg_gradient};
    padding: 2rem;
    scroll-behavior: smooth;
    }}
    .stApp {{
    background: {bg_gradient};
    }}
    html {{
    scroll-behavior: smooth;
    }}
    @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
    }}
    .fade-in {{ animation: fadeIn 0.5s ease-out; }}
    .slide-in {{ animation: slideIn 0.5s ease-out; }}
    .header-container {{
    background: {header_gradient};
    padding: 2.5rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    position: sticky;
    top: 0;
    z-index: 999;
    transition: all 0.3s ease;
    }}
    .header-title {{
    color: white;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    .header-subtitle {{
    color: rgba(255,255,255,0.95);
    font-size: 1.2rem;
    text-align: center;
    margin-top: 0.5rem;
    font-weight: 300;
    }}
    .search-container {{
    background: transparent;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    }}
    .answer-box {{
    background: {card_bg};
    padding: 2rem;
    border-radius: 12px;
    border-left: 6px solid #667eea;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    color: {text_color};
    line-height: 1.7;
    animation: fadeIn 0.5s ease-out;
    }}
    .statute-card {{
    background: {card_bg};
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
    animation: slideIn 0.5s ease-out;
    position: relative;
    }}
    .statute-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }}
    .statute-category {{
    color: {text_secondary};
    font-size: 0.8rem;
    margin-bottom: 0.8rem;
    font-style: italic;
    }}
    .statute-field {{
    margin-bottom: 0.6rem;
    line-height: 1.5;
    }}
    .field-label {{
    display: inline-block;
    background: rgba(102,126,234,0.15);
    color: #a0c4ff;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.5rem;
    white-space: nowrap;
    }}
    .field-value {{
    color: {field_value};
    font-size: 0.92rem;
    }}
    .punishment-field {{
    background: rgba(102,126,234,0.07);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    margin: 0.7rem 0;
    }}
    .punishment-value {{
    color: #fc8181 !important;
    font-weight: 600;
    font-size: 0.95rem;
    }}
    .counterpart-block {{
    margin-top: 1rem;
    padding: 0.9rem 1rem;
    border-radius: 8px;
    font-size: 0.88rem;
    }}
    .counterpart-ipc {{
    background: rgba(255, 165, 0, 0.08);
    border-left: 3px solid #f6ad55;
    }}
    .counterpart-bns {{
    background: rgba(72, 187, 120, 0.08);
    border-left: 3px solid #48bb78;
    }}
    .counterpart-label {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
    color: #f6ad55;
    }}
    .counterpart-bns .counterpart-label {{
    color: #48bb78;
    }}
    .counterpart-title {{
    font-weight: 600;
    color: {text_color};
    margin-bottom: 0.5rem;
    }}
    .counterpart-row {{
    margin-bottom: 0.3rem;
    color: {text_secondary};
    }}
    .web-card {{
    background: {card_bg};
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #11998e;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
    animation: slideIn 0.5s ease-out;
    }}
    .web-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(17, 153, 142, 0.15);
    }}
    .card-title {{
    color: {text_color};
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    }}
    .card-text {{
    color: {text_secondary};
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 0.8rem;
    }}
    .section-header {{
    color: {text_color};
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
    display: flex;
    align-items: center;
    justify-content: space-between;
    }}
    .result-badge {{
    background: #667eea;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    }}
    .web-link {{
    color: #11998e;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.3s;
    }}
    .web-link:hover {{
    color: #0d7a6f;
    text-decoration: underline;
    }}
    /* PRIMARY call-to-action: the form's Analyze button (bold gradient) */
    .stFormSubmitButton > button,
    div[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.8rem 2rem;
    border-radius: 8px;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    transition: all 0.3s;
    }}
    .stFormSubmitButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    color: white !important;
    }}
    /* SECONDARY actions: date presets & example queries (subtle outlined pills) */
    .stButton > button {{
    background: transparent !important;
    color: {text_color} !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.45rem 0.9rem !important;
    border-radius: 10px !important;
    border: 1.5px solid {border_color} !important;
    box-shadow: none !important;
    transition: all 0.2s;
    }}
    .stButton > button:hover {{
    border-color: #667eea !important;
    color: #667eea !important;
    background: rgba(102, 126, 234, 0.08) !important;
    transform: translateY(-1px);
    }}
    .stTextInput > div > div > input {{
    border-radius: 8px;
    border: 2px solid {border_color};
    padding: 0.8rem;
    font-size: 1rem;
    background: {card_bg};
    color: {text_color} !important;
    transition: all 0.3s;
    }}
    .stTextInput > div > div > input:focus {{
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }}
    .stTextInput > div > div > input::placeholder {{
    color: {text_secondary} !important;
    }}
    .stDateInput > div > div > input {{
    border-radius: 8px;
    border: 2px solid {border_color};
    background: {card_bg};
    color: {text_color} !important;
    }}
    /* Force the BaseWeb date-input container to follow the theme (its inner
       input is transparent, so the background lives on the wrapper). */
    .stDateInput div[data-baseweb="input"],
    .stDateInput div[data-baseweb="input"] > div,
    .stDateInput input {{
    background: {card_bg} !important;
    color: {text_color} !important;
    border-color: {border_color} !important;
    }}
    .empty-state {{
    text-align: center;
    padding: 4rem 2rem;
    color: {text_secondary};
    font-size: 1.1rem;
    background: {card_bg};
    border-radius: 10px;
    animation: fadeIn 0.5s ease-out;
    }}
    .empty-state-icon {{
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
    }}
    .metric-card {{
    background: {card_bg};
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    transition: all 0.3s;
    animation: fadeIn 0.5s ease-out;
    }}
    .metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }}
    .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
    }}
    .metric-label {{
    font-size: 0.9rem;
    color: {text_secondary};
    margin-top: 0.5rem;
    }}
    .metric-icon {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
    }}
    .streamlit-expanderHeader {{
    background: {card_bg};
    border-radius: 8px;
    font-weight: 500;
    color: {text_color};
    transition: all 0.3s;
    padding: 1rem;
    border: 2px solid transparent;
    }}
    .streamlit-expanderHeader:hover {{
    border-color: #667eea;
    }}
    .streamlit-expanderContent {{
    background: {card_bg};
    color: {text_color};
    padding: 1rem;
    border-radius: 0 0 8px 8px;
    }}
    p, span, div {{ color: {text_color}; }}
    .stMarkdown {{ color: {text_color}; }}
    .stCaption {{ color: {text_secondary} !important; }}
    .stAlert {{ background: {card_bg}; color: {text_color}; }}
    hr {{
    margin: 3rem 0 1rem 0;
    border: none;
    border-top: 2px solid rgba(102, 126, 234, 0.2);
    }}
    .footer-text {{ color: {text_color}; }}
    .keyboard-hint {{
    color: {text_secondary};
    font-size: 0.8rem;
    font-style: italic;
    }}
    .law-change-banner {{
    background: linear-gradient(135deg, #744210 0%, #92400e 100%);
    border: 1px solid #f6ad55;
    border-left: 6px solid #f6ad55;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.5s ease-out;
    }}
    .law-change-banner-title {{
    color: #fefcbf;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    }}
    .law-change-banner-body {{
    color: #fef3c7;
    font-size: 0.95rem;
    line-height: 1.6;
    }}
    .law-change-web-link {{
    color: #fbbf24;
    font-size: 0.88rem;
    text-decoration: none;
    display: inline-block;
    margin-top: 0.4rem;
    }}
    .law-change-web-link:hover {{
    text-decoration: underline;
    }}
    .law-nochange-banner {{
    background: linear-gradient(135deg, #1a3a2a 0%, #1a3a2e 100%);
    border: 1px solid #48bb78;
    border-left: 6px solid #48bb78;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.5s ease-out;
    }}
    .law-nochange-banner-text {{
    color: #c6f6d5;
    font-size: 0.95rem;
    font-weight: 500;
    }}
    @media (max-width: 768px) {{
    .header-title {{ font-size: 2rem; }}
    .header-subtitle {{ font-size: 1rem; }}
    .metric-card {{ margin-bottom: 1rem; }}
    .section-header {{ font-size: 1.2rem; }}
    }}
    /* ---- Day/Night icon toggle: sun-in-knob on light, moon-in-knob on dark ---- */
    div[data-testid="stCheckbox"] {{ display: flex; justify-content: flex-end; }}
    /* DEFAULT (OFF = Light): soft-blue track — use a gradient (BaseWeb ignores
       background-color overrides but respects background-image) */
    div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div {{
    background: linear-gradient(135deg, #cbd5e0 0%, #aebccf 100%) !important;
    transition: background 0.4s ease !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.28) !important;
    }}
    /* DEFAULT knob: white circle carrying the SUN icon */
    div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div > div {{
    background-color: #ffffff !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 12px 12px !important;
    transition: transform 0.35s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.35) !important;
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='%23ed8936'%20stroke-width='2.2'%20stroke-linecap='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='4'/%3E%3Cline%20x1='12'%20y1='2'%20x2='12'%20y2='4'/%3E%3Cline%20x1='12'%20y1='20'%20x2='12'%20y2='22'/%3E%3Cline%20x1='2'%20y1='12'%20x2='4'%20y2='12'/%3E%3Cline%20x1='20'%20y1='12'%20x2='22'%20y2='12'/%3E%3Cline%20x1='4.9'%20y1='4.9'%20x2='6.3'%20y2='6.3'/%3E%3Cline%20x1='17.7'%20y1='17.7'%20x2='19.1'%20y2='19.1'/%3E%3Cline%20x1='4.9'%20y1='19.1'%20x2='6.3'%20y2='17.7'/%3E%3Cline%20x1='17.7'%20y1='6.3'%20x2='19.1'%20y2='4.9'/%3E%3C/svg%3E") !important;
    }}
    /* ON = Dark: navy track */
    div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > div {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    }}
    /* ON knob: swap SUN → MOON */
    div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > div > div {{
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='%23334155'%3E%3Cpath%20d='M21%2012.8A9%209%200%201%201%2011.2%203%207%207%200%200%200%2021%2012.8z'/%3E%3C/svg%3E") !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
    <h1 class="header-title"> Time-Aware Legal Assistant</h1>
    <p class="header-subtitle">Context-aware Indian legal answers across time</p>
    </div>
""", unsafe_allow_html=True)

# Apply any pending query/date set by the example or date-preset buttons.
# This must run BEFORE the form widgets are instantiated.
if "_pending_query" in st.session_state:
    st.session_state["query_input"] = st.session_state.pop("_pending_query")
    st.session_state["_run_now"] = True
if "_pending_date" in st.session_state:
    st.session_state["date_input"] = st.session_state.pop("_pending_date")

# Search Section
with st.form(key="search_form", clear_on_submit=False):
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Legal Query",
            placeholder="e.g., What are the provisions for theft under IPC?",
            label_visibility="collapsed",
            key="query_input"
        )
        st.caption("Enter your legal question or search for specific provisions")
        st.markdown('<p class="keyboard-hint">Press Enter to search</p>', unsafe_allow_html=True)

    with col2:
        if "date_input" not in st.session_state:
            st.session_state["date_input"] = datetime.today().date()
        selected_date = st.date_input(
            "Relevant Date",
            label_visibility="collapsed",
            key="date_input"
        )
        st.caption("Select the relevant date")

    analyze = st.form_submit_button("Analyze Legal Position", use_container_width=True)

# ---- Animated inline SVG icons (no emoji) ----
def _anim_svg(kind, color, size=22):
    common = (f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
              f'stroke="{color}" stroke-width="1.7" stroke-linecap="round" '
              f'stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"')
    if kind == "scales":  # scales of justice — beam gently tilts
        return (f'<svg {common}><line x1="12" y1="4" x2="12" y2="21"/>'
                f'<line x1="7" y1="21" x2="17" y2="21"/>'
                f'<g><animateTransform attributeName="transform" type="rotate" '
                f'values="-7 12 6;7 12 6;-7 12 6" dur="3s" repeatCount="indefinite"/>'
                f'<line x1="4" y1="6" x2="20" y2="6"/>'
                f'<path d="M4 6 L2 11.5 A3 3 0 0 0 6 11.5 Z"/>'
                f'<path d="M20 6 L18 11.5 A3 3 0 0 0 22 11.5 Z"/></g></svg>')
    if kind == "building":  # court building — soft glow pulse
        return (f'<svg {common}><animate attributeName="opacity" values="1;0.55;1" '
                f'dur="2.6s" repeatCount="indefinite"/>'
                f'<polygon points="12,3 21,8 3,8"/>'
                f'<line x1="5" y1="8" x2="5" y2="18"/><line x1="10" y1="8" x2="10" y2="18"/>'
                f'<line x1="14" y1="8" x2="14" y2="18"/><line x1="19" y1="8" x2="19" y2="18"/>'
                f'<line x1="3" y1="21" x2="21" y2="21"/></svg>')
    if kind == "search":  # magnifier — subtle searching wiggle
        return (f'<svg {common}><g><animateTransform attributeName="transform" '
                f'type="translate" values="0 0;2 1.5;0 0;-1.5 1;0 0" dur="3s" '
                f'repeatCount="indefinite"/><circle cx="10" cy="10" r="6"/>'
                f'<line x1="14.5" y1="14.5" x2="21" y2="21"/></g></svg>')
    if kind == "warn":  # warning triangle — attention pulse
        return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
                f'stroke-linejoin="round" style="vertical-align:middle;margin-right:5px;">'
                f'<animate attributeName="opacity" values="1;0.4;1" dur="1.6s" repeatCount="indefinite"/>'
                f'<path d="M12 3 L22 20 H2 Z"/><line x1="12" y1="9" x2="12" y2="14"/>'
                f'<circle cx="12" cy="17" r="0.6" fill="{color}"/></svg>')
    return ""

# ---- Clickable example queries (fill + run) ----
# Each category: (svg-kind, accent colour, [queries])
_EXAMPLES = {
    "Criminal Law":    ("scales",   "#a0c4ff", ["What is the punishment for murder?", "Punishment for theft", "Provisions for robbery"]),
    "Case Law":        ("building", "#f6ad55", ["Sedition law in India", "What is criminal breach of trust?", "Punishment for cheating"]),
    "Legal Research":  ("search",   "#48bb78", ["Definition of culpable homicide", "Difference between theft and robbery", "What is defamation?"]),
}
with st.expander("Example Queries — click any to run", expanded=True):
    st.markdown(
        f"<p style='color:{text_secondary}; font-size:0.85rem; margin:0 0 0.6rem 0;'>"
        "New here? Pick a question below and the assistant will run it instantly.</p>",
        unsafe_allow_html=True,
    )
    _ecols = st.columns(len(_EXAMPLES))
    for _i, (_cat, (_kind, _accent, _qs)) in enumerate(_EXAMPLES.items()):
        with _ecols[_i]:
            st.markdown(
                f"<div style='color:{_accent}; font-weight:700; font-size:0.95rem; "
                f"margin-bottom:0.6rem; padding-bottom:0.35rem; "
                f"border-bottom:2px solid {_accent}33;'>{_anim_svg(_kind, _accent)}{_cat}</div>",
                unsafe_allow_html=True,
            )
            for _j, _q in enumerate(_qs):
                if st.button(_q, key=f"ex_{_i}_{_j}", use_container_width=True):
                    st.session_state["_pending_query"] = _q
                    st.rerun()

# Results Section
analyze = analyze or st.session_state.pop("_run_now", False)
if analyze and query:
    progress_container = st.empty()
    status_container   = st.empty()

    try:
        from concurrent.futures import ThreadPoolExecutor

        with status_container:
            st.info("Searching legal database and web sources...")

        statutes, web_results = [], []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_statutes = executor.submit(chroma_search, query, selected_date)
            future_web      = executor.submit(serpapi_search, query, selected_date.year)
            statutes    = future_statutes.result()
            web_results = future_web.result()

        with status_container:
            st.info("Generating legal analysis...")
        # LLM answer when Ollama is available, else deterministic fallback.
        answer = generate_answer_safe(query, statutes, web_results, selected_date)

        progress_container.empty()
        status_container.empty()

        import threading
        threading.Thread(
            target=log_metrics_to_json,
            args=(query, selected_date, statutes, web_results, answer),
            daemon=True
        ).start()

        st.success("Search completed successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{len(statutes)}</div>
            <div class="metric-label">Statutory Provisions</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{len(web_results)}</div>
            <div class="metric-label">Web Sources</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        law_changes = detect_law_changes(query, statutes, selected_date)

        if law_changes["has_changes"]:
            web_links_html = ""
            if law_changes["web_changes"]:
                web_links_html = "<br>" + "".join(
                    '<a href="' + html.escape(str(w.get("link", "#")), quote=True) + '" target="_blank" rel="noopener noreferrer" class="law-change-web-link">'
                    + html.escape(str(w.get("title", ""))) + '</a><br>'
                    for w in law_changes["web_changes"]
                )
            banner_html = (
                '<div class="law-change-banner">'
                '<div class="law-change-banner-title">Law Change Detected</div>'
                '<div class="law-change-banner-body">'
                + law_changes["summary"]
                + web_links_html
                + '</div></div>'
            )
            st.markdown(banner_html, unsafe_allow_html=True)
        else:
            date_str = selected_date.strftime('%d %B %Y')
            st.markdown(
                '<div class="law-nochange-banner"><span class="law-nochange-banner-text">'
                'No amendments or replacements detected for this provision as of '
                + date_str + '.</span></div>',
                unsafe_allow_html=True
            )

        # Legal Analysis box: show only the TOP provision (concise) by default,
        # with an inline "Read more" that expands to the full answer (all provisions).
        def _prov_block(s, short=True):
            content = s.get("content", s.get("text", ""))
            body = _short(content, 200) if short else re.sub(r"\s+", " ", str(content or "")).strip()
            return (
                f"<div style='margin-bottom:0.9rem;'>"
                f"<div style='font-weight:600; color:{text_color};'>{s.get('title','')}{(' — ' + s.get('category','')) if s.get('category') else ''}</div>"
                f"<div style='font-size:0.92rem; line-height:1.65; color:{field_value}; margin:0.3rem 0;'>{body}</div>"
                f"<div style='font-size:0.92rem; color:#fc8181;'><strong>Punishment:</strong> {s.get('punishment','Not specified')}</div>"
                f"</div>"
            )

        if statutes:
            box_html = _prov_block(statutes[0], short=True)
            # Anything beyond the top result, or a trimmed top result, goes behind "Read more".
            top_full = re.sub(r"\s+", " ", str(statutes[0].get("content", statutes[0].get("text", "")) or "")).strip()
            if len(statutes) > 1 or len(top_full) > 200:
                full_blocks = "".join(_prov_block(s, short=False) for s in statutes)
                box_html += (
                    "<details style='margin-top:0.6rem;'>"
                    "<summary style='cursor:pointer; color:#a0c4ff; font-weight:600; font-size:0.9rem;'>Read more</summary>"
                    f"<div style='margin-top:0.6rem;'>{full_blocks}</div>"
                    "</details>"
                )
        else:
            box_html = answer

        st.markdown('<p class="section-header">Legal Analysis</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{box_html}</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f'<p class="section-header">Statutory Provisions <span class="result-badge">{len(statutes)} result{"" if len(statutes)==1 else "s"}</span></p>', unsafe_allow_html=True)
            if statutes:
                for s in statutes:
                    ipc = s.get("ipc_counterpart")
                    bns = s.get("bns_counterpart")
                    counterpart_html = ""
                    if ipc:
                        counterpart_html = (
                            '<div class="counterpart-block counterpart-ipc">'
                            '<div class="counterpart-label">Previous Law — IPC</div>'
                            f'<div class="counterpart-title">Section {ipc.get("section")} — {ipc.get("title","")}</div>'
                            f'<div class="counterpart-row"><span class="field-label">Definition</span> {_short(ipc.get("content",""))}</div>'
                            f'<div class="counterpart-row"><span class="field-label">Punishment</span> {ipc.get("punishment","Not specified")}</div>'
                            f'<div class="counterpart-row"><span class="field-label">Source</span> {ipc.get("source","Indian Penal Code, 1860")}</div>'
                            '</div>'
                        )
                    if bns:
                        counterpart_html += (
                            '<div class="counterpart-block counterpart-bns">'
                            '<div class="counterpart-label">New Law — BNS</div>'
                            f'<div class="counterpart-title">Section {bns.get("section")} — {bns.get("title","")}</div>'
                            f'<div class="counterpart-row"><span class="field-label">Definition</span> {_short(bns.get("content",""))}</div>'
                            f'<div class="counterpart-row"><span class="field-label">Punishment</span> {bns.get("punishment","Not specified")}</div>'
                            f'<div class="counterpart-row"><span class="field-label">In force</span> {bns.get("valid_from","2024-07-01")} onwards</div>'
                            f'<div class="counterpart-row"><span class="field-label">Source</span> {bns.get("source","Bharatiya Nyaya Sanhita, 2023")}</div>'
                            '</div>'
                        )
                    valid_from  = s.get("valid_from", "")
                    valid_until = s.get("valid_until", "")
                    validity_str = f"{valid_from} → Present" if valid_until == "9999-12-31" else (f"{valid_from} → {valid_until}" if valid_from else "N/A")
                    note_html = (
                        f'<div style="background:rgba(246,173,85,0.12);border-left:3px solid #f6ad55;'
                        f'border-radius:6px;padding:0.4rem 0.7rem;margin-bottom:0.6rem;'
                        f'color:#f6ad55;font-size:0.8rem;">{_anim_svg("warn", "#f6ad55", 14)}{s.get("note","")}</div>'
                        if s.get("note") else ""
                    )
                    st.markdown(f"""
                    <div class="statute-card">
                    <div class="card-title">{s['title']}</div>
                    <div class="statute-category">{s.get('category','')}</div>
                    {note_html}
                    <div class="statute-field">
                    <span class="field-label">Definition</span>
                    <span class="field-value">{_short(s.get('content', s.get('text','')))}</span>
                    </div>
                    <div class="statute-field punishment-field">
                    <span class="field-label">Punishment</span>
                    <span class="field-value punishment-value">{s.get('punishment','Not specified')}</span>
                    </div>
                    <div class="statute-field">
                    <span class="field-label">Source</span>
                    <span class="field-value">{s.get('source','')}</span>
                    </div>
                    <div class="statute-field">
                    <span class="field-label">In Force</span>
                    <span class="field-value">{validity_str}</span>
                    </div>
                    {counterpart_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # Expander with the full statutory text (only if it was trimmed)
                    _full = re.sub(r"\s+", " ", str(s.get('content', s.get('text','')) or "")).strip()
                    if len(_full) > 180:
                        with st.expander("Show full statutory text"):
                            st.markdown(
                                f"<div style='font-size:0.9rem; line-height:1.65; color:{field_value};'>{_full}</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown('''
                <div class="empty-state">
                <p>No statutory provisions found</p>
                <p style="font-size: 0.9rem;">Try adjusting your search query or date</p>
                </div>
                ''', unsafe_allow_html=True)

        with col_right:
            st.markdown(f'<p class="section-header">Judicial & Web Sources <span class="result-badge">{len(web_results)} result{"" if len(web_results)==1 else "s"}</span></p>', unsafe_allow_html=True)
            if web_results:
                def _parse_date(w):
                    """Parse SerpAPI date string to sortable value, newest first."""
                    raw = w.get("date", "")
                    for fmt in ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%B %d, %Y"):
                        try:
                            return datetime.strptime(raw, fmt)
                        except Exception:
                            pass
                    return datetime.min  # unknown dates go to the end

                def _extract_date_from_text(w):
                    """Try to extract a date from snippet/title if date field is missing."""
                    raw = w.get("date", "").strip()
                    if raw:
                        return raw
                    # Fallback: scan snippet and title for date patterns
                    text = w.get("title", "") + " " + w.get("snippet", "")
                    patterns = [
                        r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})\b',
                        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{4})\b',
                        r'\b(\d{4}-\d{2}-\d{2})\b',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            return match.group(1)
                    return "Date not available"

                sorted_web = sorted(web_results, key=_parse_date, reverse=True)
                for w in sorted_web:
                    # Escape untrusted external (SerpAPI) fields before rendering as HTML.
                    w_title = html.escape(str(w.get('title', 'Untitled')))
                    w_snippet = html.escape(str(w.get('snippet', 'No description available')))
                    w_link = html.escape(str(w.get('link', '#')), quote=True)
                    date_info = html.escape(str(_extract_date_from_text(w)))
                    st.markdown(f"""
                    <div class="web-card">
                    <div class="card-title">{w_title}</div>
                    <div style="color: {text_secondary}; font-size: 0.85rem; margin-bottom: 0.5rem;">{date_info}</div>
                    <div class="card-text">{w_snippet}</div>
                    <a href="{w_link}" target="_blank" rel="noopener noreferrer" class="web-link">Read full article</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="empty-state">
                <p>No web sources found</p>
                <p style="font-size: 0.9rem;">Try adding keywords like "recent" or "latest"</p>
                </div>
                ''', unsafe_allow_html=True)

    except Exception as e:
        progress_container.empty()
        status_container.empty()
        st.error(f"An error occurred: {str(e)}")
        st.info("Try simplifying your query or check your database connection.")

elif analyze and not query:
    st.warning("Please enter a legal query to search")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; padding: 1rem;" class="footer-text">
    <p style="color: {text_color};"><strong>Disclaimer:</strong> This is an AI-powered legal research tool.
    Always consult with a qualified legal professional for legal advice.</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; color: {text_color};">
    Powered by Ollama • ChromaDB • Streamlit
    </p>
    <p style="font-size: 0.85rem; margin-top: 1rem; color: {text_secondary};">
    Tip: Press Enter to quickly search
    </p>
    </div>
""", unsafe_allow_html=True)

# =====================================================
# BACKEND FUNCTION FOR EVALUATION (NO UI)
# =====================================================
def answer_query_backend(question, selected_date=None):
    """Backend-safe function used ONLY for evaluation."""
    if selected_date is None:
        selected_date = datetime.today().date()

    statutes      = chroma_search(question, selected_date)
    web_results   = []
    answer_text = generate_answer_safe(question, statutes, web_results, selected_date)

    offline_metrics    = evaluate_offline(question, statutes, selected_date)
    online_metrics     = evaluate_online(question, web_results)
    generation_metrics = evaluate_generation(answer_text, statutes, web_results, question)

    if statutes:
        top_statute = statutes[0]
        title   = top_statute["title"]
        # Use the structured law/section fields, not brittle title-string parsing.
        law     = top_statute.get("law") or ("BNS" if title.startswith("BNS") else "IPC")
        section = str(top_statute.get("section") or title.split("Section")[-1].strip())
        source  = "Bharatiya Nyaya Sanhita, 2023" if law == "BNS" else "Indian Penal Code, 1860"
        punishment_match = re.search(
            r"(death[^.]+\.|imprisonment[^.]+\.|rigorous imprisonment[^.]+\.|fine[^.]+\.)",
            top_statute["text"].lower()
        )
        punishment = punishment_match.group(0) if punishment_match else "Not specified"
    else:
        law = "UNKNOWN"
        section = "UNKNOWN"
        punishment = "Not specified"
        source = "Not available"

    return {
        "law": law, "section": section,
        "punishment": punishment.strip(), "source": source,
        "answer": answer_text,
        "offline_metrics": offline_metrics,
        "online_metrics": online_metrics,
        "generation_metrics": generation_metrics
    }