# Known Issues & Retrieval Review

Findings from a review of the retrieval logic in `app.py`. Ordered by severity.
Line numbers refer to `app.py` at the time of writing.

## 1. Temporal override ignores the selected date (HIGH) — ✅ FIXED

`rule_based_override()` (app.py:327) runs *after* the date-aware boost in `chroma_search()`
and hardcodes a BNS target for topic keywords (e.g. `"murder" → ("BNS", "101")`), regardless
of the selected date. The confirmed result:

```
query "punishment for murder", date 2020-01-01  ->  BNS 101   (WRONG)
```

In 2020 the IPC was in force, so this should be **IPC 302**. The whole selling point of the
project — temporal accuracy — is defeated for any keyword that has a `RULE_MAP` entry.

**Fixed:** `rule_based_override(query, statutes, qdate)` now selects `ipc_target` vs
`bns_target` by an explicit law in the query first, then by the `BNS_CUTOFF` (20240701) date.
Verified: `"murder"` @ 2020 → **IPC 302**, @ 2025 → **BNS 101**, `"murder under IPC"` @ 2025 →
**IPC 302**.

## 2. Bare-section lookup collides IPC and BNS (HIGH) — ✅ FIXED

In `_build_full_index()` the index is written twice per entry (app.py:103–106):
`section_map[f"{law}_{sec}"]` **and** `section_map[sec]`. Because data loads IPC first then BNS,
the bare key is overwritten — `SECTION_MAP["304"]` resolves to **BNS**, not IPC. The direct
lookup in `chroma_search()` (app.py:500) uses the bare section number and ignores both the law
named in the query and the date:

```
query "IPC Section 304", date 2020-01-01  ->  BNS 304   (WRONG law and WRONG era)
```

**Fixed:** the direct-lookup branch in `chroma_search` now reads the law-qualified
`IPC_{sec}` / `BNS_{sec}` keys and picks by explicit law in the query, else by date.
Verified: `"IPC Section 304"` @ 2020 → **IPC 304**, `"BNS Section 304"` @ 2025 → **BNS 304**,
bare `"Section 304"` follows the date, `"IPC Section 999"` → empty (no hallucination).

## 3. Retrieval is steered by hand-tuned tables, not semantics (MEDIUM) — ✅ FIXED

Ranking depends on four overlapping hardcoded tables: `_SECTION_SYNONYMS` / `_IPC_SYNONYMS`
(with words repeated 8–12× to inflate TF-IDF weight — app.py:49, 120), `RULE_MAP` (app.py:335),
and `KEYWORD_SECTION_MAP` (app.py:405). This works only for the ~20 curated queries it was tuned
against and does not generalise; every new query class needs another hand-written rule. The
imported ChromaDB/Ollama stack (which would give real semantic retrieval) is unused on the live
path.

**Fixed:** added a local `sentence-transformers` (all-MiniLM-L6-v2) semantic index
(`get_semantic_index`) and a hybrid ranker `_hybrid_score_all()` = `0.6*cosine + 0.4*tfidf_norm`,
now the primary path in `chroma_search`. The TF-IDF weight-inflation hack (synonyms repeated
×8–12) was removed (synonyms de-duplicated); the rule tables remain only as a thin, date-aware
post-filter. Verified generalisation to phrasings with **no** rule-table entry — e.g.
"someone poisoned my food to kill me" → murder (BNS 103), "stealing a mobile phone" → theft,
"setting fire to a house" → BNS 305 — which the old synonym-only approach could not do.
Note: the stored ChromaDB vectors are 768-dim `nomic-embed-text` (Ollama) and are unusable
without that model, so the local MiniLM index is used instead. Falls back to pure TF-IDF if the
model can't load.

## 4. Evaluation metrics are near-tautological (MEDIUM) — ✅ FIXED

`evaluate_offline()` computed `Precision = len(statutes)/len(statutes) = 1.0` whenever any result
was returned, and `Temporal Accuracy` was hardcoded to `1.0` — so it could not detect the
failures in issues #1 and #2.

**Fixed:** `evaluate_offline()` now reports four honestly-computable signals:
`Retrieval Confidence` (normalised top score), `Score Margin` (#1 vs #2 separation),
`Temporal Validity` (fraction of results actually in force on the selected date), and
`Top-K Coverage`. **Temporal Validity is the key one** — it recomputes each result's
`valid_from`/`valid_until` against the date, so a temporal regression (returning out-of-era law)
now drops the score below 1.0 instead of silently reporting a perfect 1.0. Empty results score 0
across the board. For labelled accuracy, the standalone `evaluation.py` (checked against
`ground_truth.json`) is still the reference.

## 5. Minor

- `filter_hallucination()` (app.py:821) only matches `section <number>` and misses sections with
  sub-parts like `103(2)`, so a valid lynching answer could be wrongly rejected.
- ✅ FIXED — `answer_query_backend()` now reads the structured `law`/`section` fields instead of
  `title.startswith("BNS")` / title string-splitting.
- ✅ FIXED (evaluation.py) — `detect_hallucination()` checked the detected punishment against the
  `text`/definition field only, so real punishments were falsely flagged as hallucinated
  (inflating the rate to ~72%). It now checks the `punishment` field too; on the 40-query set the
  measured hallucination rate dropped to 2.5%.
- ✅ ADDED — new placeholder sections now show a "pending ingestion" badge in the statute card.
- `SERPAPI_KEY` defaults to empty, so web search and `detect_law_changes` web calls silently
  no-op without a key — expected, but worth documenting (now in the README).
