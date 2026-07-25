import json, math, re
from collections import defaultdict

STOP_WORDS = {
    "a","an","the","of","by","or","and","to","in","for","with","on",
    "at","from","not","amounting","attempt","commit","causing",
    "voluntarily","belonging","what","is","are","under","give","me",
    "tell","explain","describe","provide","recent","latest",
}

SYNONYMS = {
    "152": ["sedition"] * 8 + ["seditious", "disaffection", "hatred", "contempt", "state", "offences"],
    "103": ["murder"] * 8 + ["homicide", "killing", "kill", "death", "intentional"],
    "105": ["culpable", "homicide", "culpable homicide"] * 5 + ["death", "bodily", "injury"],
    "106": ["negligence"] * 6 + ["negligent", "rash", "accident", "accidental", "death"],
    "63":  ["rape"] * 8 + ["sexual", "assault", "consent", "intercourse"],
    "64":  ["rape"] * 6 + ["minor", "sixteen", "child", "underage"],
    "303": ["theft"] * 8 + ["stealing", "stolen", "movable", "property"],
    "304": ["theft"] * 6 + ["stealing", "punishment"],
    "309": ["robbery"] * 8 + ["robbed", "force", "hurt"],
    "310": ["dacoity"] * 8 + ["dacoits", "gang", "robbery"],
    "137": ["kidnap"] * 8 + ["kidnapping", "abduction", "abduct"],
    "318": ["cheating"] * 8 + ["fraud", "deceive", "deception", "dishonest"],
}

def tokenise(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower())
            if w not in STOP_WORDS and len(w) > 2]

data = []
for path in ["IPC_updated.json", "BNS_updated.json"]:
    with open(path, encoding="utf-8") as f:
        data.extend(json.load(f))

docs = {}
section_map = {}
for entry in data:
    sec = str(entry.get("section","")).strip()
    law = entry.get("law","")
    key = f"{law}_{sec}"
    section_map[key] = entry
    section_map[sec] = entry
    synonyms = " ".join(SYNONYMS.get(sec, [])) if law == "BNS" else ""
    full_text = f"{sec} {law} {entry.get('title','')} {entry.get('category','')} {entry.get('content','')} {entry.get('punishment','')} {synonyms}"
    docs[key] = tokenise(full_text)

df = defaultdict(int)
for tokens in docs.values():
    for t in set(tokens): df[t] += 1
N = len(docs)
idf = {t: math.log((N+1)/(f+1))+1 for t,f in df.items()}
tfidf = {}
for key, tokens in docs.items():
    tf = defaultdict(int)
    for t in tokens: tf[t] += 1
    tfidf[key] = {t:(c/len(tokens))*idf.get(t,1) for t,c in tf.items()}
inverted = defaultdict(list)
for key, weights in tfidf.items():
    for term, weight in weights.items():
        inverted[term].append((key, weight))

for query in ["what is the punishment for sedition", "punishment for murder", "theft punishment"]:
    print(f"\nQuery: '{query}'")
    q_tokens = tokenise(query)
    q_tf = defaultdict(int)
    for t in q_tokens: q_tf[t] += 1
    q_tfidf = {t:(c/len(q_tokens))*idf.get(t,1) for t,c in q_tf.items()}
    scores = defaultdict(float)
    for term, q_w in q_tfidf.items():
        for key, d_w in inverted.get(term, []):
            scores[key] += q_w * d_w

    results = [(s,k) for k,s in scores.items() if s > 0 and "_" in k]
    results.sort(reverse=True)
    qdate = 20260313
    print(f"  {'Key':<15} {'Boosted':>10} Status")
    for score, key in results[:5]:
        entry = section_map.get(key, {})
        vf = int(entry.get("valid_from","0000-00-00").replace("-",""))
        vu = int(entry.get("valid_until","9999-12-31").replace("-",""))
        active = vf <= qdate <= vu
        boosted = score*3.0 if active else score
        status = "✅ ACTIVE" if active else "❌ EXPIRED"
        print(f"  {key:<15} {boosted:>10.4f}  {status}")