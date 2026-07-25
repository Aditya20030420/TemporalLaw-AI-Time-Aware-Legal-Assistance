# Legal Assistant Evaluation System

## Overview
This evaluation module measures the performance of your time-aware legal chatbot on 5 key metrics specific to Indian legal research.

## Metrics Explained

### 1. Answer Correctness (%)
**What it measures:** Whether the generated answer contains the correct section number and punishment.

**How it works:**
- Compares detected section number with expected section
- Checks if punishment keywords match
- Answer is correct if EITHER section OR punishment matches

**Interpretation:**
- ≥ 70% = Good performance
- < 70% = Review retrieval quality or answer generation logic

---

### 2. Temporal Accuracy (%) ⭐
**What it measures:** Whether the system prefers BNS (2024 law) over IPC when applicable.

**How it works:**
- For queries about current law (2024+), expects BNS
- For queries about old law (pre-2024), expects IPC
- Marks temporal violation if BNS is expected but IPC is returned

**Interpretation:**
- ≥ 70% = System correctly handles time-aware retrieval
- < 70% = Date filtering or BNS prioritization needs improvement

**Academic Justification:**
This metric is critical for legal systems because:
- Laws change over time (IPC replaced by BNS in 2024)
- Providing outdated legal information can have serious consequences
- Temporal awareness is a unique challenge in legal RAG systems

---

### 3. Law Selection Accuracy (%)
**What it measures:** Whether the correct law family (IPC vs BNS) was identified.

**How it works:**
- Extracts law name from generated answer
- Compares with expected law (IPC or BNS)

**Interpretation:**
- ≥ 70% = Good law detection
- < 70% = Improve law name extraction or embedding quality

---

### 4. Citation Validity (%)
**What it measures:** Whether citations are complete and verifiable.

**How it works:**
- Checks if law name is present (IPC or BNS)
- Checks if section number is present
- Verifies section exists in retrieved documents (prevents hallucination)

**Interpretation:**
- ≥ 70% = Citations are reliable
- < 70% = Improve prompt engineering to enforce citation

---

### 5. Hallucination Rate (%)
**What it measures:** How often the model generates false or ungrounded information.

**How it works:**
Marks hallucination if:
- Section number is mentioned but doesn't exist in retrieved statutes
- Punishment details are fabricated (not found in sources)
- Answer is generated without any retrieved context

**Interpretation:**
- < 20% = Low hallucination (good)
- ≥ 20% = High hallucination (needs improvement)

**How to reduce hallucination:**
- Improve retrieval relevance (better embeddings)
- Add stricter grounding in system prompt
- Increase context window with more retrieved documents

---

## Usage

### Basic Usage
```bash
python evaluation.py
```

This will:
1. Load `test_questions.json`
2. Run each question through your RAG pipeline
3. Calculate all 5 metrics
4. Display results in a table
5. Save results to `evaluation_results.json`

### Custom Dataset
```bash
python evaluation.py --dataset my_test_cases.json
```

---

## Dataset Format

### JSON Format (Recommended)
```json
[
  {
    "question": "What is the punishment for murder under Indian law?",
    "expected_law": "BNS",
    "expected_section": "103",
    "expected_punishment": "death or life imprisonment",
    "valid_date": "2024-07-01"
  }
]
```

### CSV Format
```csv
question,expected_law,expected_section,expected_punishment,valid_date
What is the punishment for murder?,BNS,103,death or life imprisonment,2024-07-01
```

---

## Example Output

```
================================================================================
EVALUATION RESULTS
================================================================================

┌─────────────────────────────────┬──────────┐
│ Metric                          │ Score    │
├─────────────────────────────────┼──────────┤
│ ✓ Answer Accuracy (%)           │  85.00%  │
│ ✓ Temporal Accuracy (%)         │  90.00%  │
│ ✓ Law Selection Accuracy (%)    │  88.00%  │
│ ✓ Citation Validity (%)         │  92.00%  │
│ ✓ Hallucination Rate (%)        │  10.00%  │
└─────────────────────────────────┴──────────┘

INTERPRETATION:
---------------
✓ System is performing well across all metrics.
  Temporal accuracy is strong - BNS is correctly preferred over IPC.

================================================================================
```

---

## Integration with Existing Code

The evaluation module imports from your existing `legal_assistant_ultimate.py`:
- `chroma_search()` - Retrieves statutes
- `generate_answer()` - Generates RAG-based answer
- `serpapi_search()` - Optional web search

**No modifications needed to your existing code!**

---

## Academic Context

### Why These Metrics?

**Traditional NLP metrics (BLEU, ROUGE, F1) are insufficient because:**
1. Legal text requires exact precision (word-level similarity isn't enough)
2. Temporal awareness has no equivalent in standard NLP tasks
3. Citation validity is domain-specific
4. Hallucination in legal context has serious real-world consequences

**Our metrics are designed for:**
- Final year engineering project evaluation
- Legal domain-specific requirements
- Explainable and auditable results
- Academic defense of methodology

---

## Troubleshooting

### "Dataset file not found"
- Run `python evaluation.py` first
- It will auto-create `test_questions.json` with examples
- Add your own test cases

### "Could not import from legal_assistant_ultimate"
- Make sure `evaluation.py` is in the same directory as your main code
- Check that your main file is named `legal_assistant_ultimate.py`
- Adjust import statements if needed

### Low Temporal Accuracy
- Verify ChromaDB contains BNS laws (post-2024)
- Check date filtering in `chroma_search()`
- Ensure BNS embeddings are present

### High Hallucination Rate
- Improve retrieval (better embeddings, more data)
- Strengthen grounding in system prompt
- Add explicit "cite your sources" instruction

---

## Extension Ideas

1. **Add more test cases** - Cover edge cases and rare sections
2. **Per-category metrics** - Separate scores for murder, theft, etc.
3. **Confidence scoring** - Track when model is uncertain
4. **Comparative analysis** - Test different embedding models

---

## Citation

If you use this evaluation framework in your project report:

```
Time-Aware Legal RAG Evaluation Framework
Metrics: Answer Correctness, Temporal Accuracy, Law Selection, Citation Validity, Hallucination Rate
Designed for IPC/BNS Indian Legal System
```

---

## Contact & Support

For questions about the evaluation logic or metrics:
- Review comments in `evaluation.py`
- Check individual metric calculation functions
- Refer to this README

**Good luck with your final year project! 🎓**
