import json, math, re
from collections import defaultdict

data = []
for path in ["IPC_updated.json", "BNS_updated.json"]:
    with open(path, encoding="utf-8") as f:
        data.extend(json.load(f))

STOP_WORDS = {
    "a","an","the","of","by","or","and","to","in","for","with","on",
    "at","from","not","amounting","attempt","commit","causing",
    "voluntarily","belonging","what","is","are","under","give","me",
    "tell","explain","describe","provide","recent","latest",
}

def tokenise(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower())
            if w not in STOP_WORDS and len(w) > 2]

docs = {}
section_map = {}
for entry in data:
    sec = str(entry.get("section","")).strip()
    docs[sec] = tokenise(
        f"{sec} {entry.get('title','')} {entry.get('category','')} "
        f"{entry.get('content','')} {entry.get('punishment','')}"
    )
    section_map[sec] = entry

df = defaultdict(int)
for tokens in docs.values():
    for t in set(tokens): df[t] += 1

N = len(docs)
idf = {t: math.log((N+1)/(f+1))+1 for t,f in df.items()}

tfidf = {}
for sec, tokens in docs.items():
    tf = defaultdict(int)
    for t in tokens: tf[t] += 1
    tfidf[sec] = {t:(c/len(tokens))*idf.get(t,1) for t,c in tf.items()}

inverted = defaultdict(list)
for sec, weights in tfidf.items():
    for term, weight in weights.items():
        inverted[term].append((sec, weight))

query = "what is the punishment for sedition"
q_tokens = tokenise(query)
print(f"Query tokens after stop word filter: {q_tokens}\n")

if not q_tokens:
    print("❌ ALL TOKENS FILTERED — query collapses to nothing!")
    print("This is why search fails — fix the stop words.")
else:
    q_tf = defaultdict(int)
    for t in q_tokens: q_tf[t] += 1
    q_tfidf = {t:(c/len(q_tokens))*idf.get(t,1) for t,c in q_tf.items()}

    scores = defaultdict(float)
    for term, q_w in q_tfidf.items():
        for sec, d_w in inverted.get(term, []):
            scores[sec] += q_w * d_w

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    qdate = 20260313

    print("Top 10 TF-IDF scores (raw vs boosted):")
    print(f"{'Section':<12} {'Law':<5} {'Raw':>8} {'Boosted':>10} {'Status'}")
    print("-" * 55)
    for sec, score in results:
        entry = section_map[sec]
        vf = int(entry.get("valid_from","0000-00-00").replace("-",""))
        vu = int(entry.get("valid_until","9999-12-31").replace("-",""))
        active = vf <= qdate <= vu
        boosted = score * 3.0 if active else (score * 0.5 if vf > qdate else score)
        status = "✅ ACTIVE" if active else ("❌ EXPIRED" if vu < qdate else "⏳ FUTURE")
        print(f"{sec:<12} {entry.get('law',''):<5} {score:>8.4f} {boosted:>10.4f}  {status}")

    print()
    print("IPC 124A content keywords:", tokenise(section_map.get("124A",{}).get("content","")))
    print("BNS 152 content keywords: ", tokenise(section_map.get("152",{}).get("content","")))