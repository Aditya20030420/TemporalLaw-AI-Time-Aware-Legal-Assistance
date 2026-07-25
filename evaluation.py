"""
Evaluation Module for Time-Aware Legal Assistant
=================================================
Measures system performance on IPC/BNS law retrieval and answer generation.

Metrics:
1. Answer Correctness - Does the answer match expected section and punishment?
2. Temporal Accuracy - Does the system prefer BNS (2024) over IPC when applicable?
3. Law Selection Accuracy - Is the correct law family (IPC/BNS) selected?
4. Citation Validity - Are citations complete and verifiable?
5. Hallucination Rate - Does the model generate false information?
"""

import json
import csv
import re
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Import your existing RAG components
# Adjust these imports based on your actual file structure
try:
    from app import (
        chroma_search,
        generate_answer,
        serpapi_search,
        collection
    )
except ImportError:
    print("Error: Could not import from app.py")
    print("Make sure this file is in the same directory.")
    sys.exit(1)


# =====================================================
# EVALUATION DATASET LOADER
# =====================================================

def load_evaluation_dataset(filepath: str) -> List[Dict]:
    """
    Load test questions from CSV or JSON file.
    
    Expected format (CSV):
    question,expected_law,expected_section,expected_punishment,valid_date
    
    Expected format (JSON):
    [
        {
            "question": "What is punishment for murder?",
            "expected_law": "BNS",
            "expected_section": "103",
            "expected_punishment": "death or life imprisonment",
            "valid_date": "2024-07-01"
        }
    ]
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif filepath.endswith('.csv'):
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({
                    'question': row['question'],
                    'expected_law': row['expected_law'].upper(),
                    'expected_section': row['expected_section'],
                    'expected_punishment': row['expected_punishment'].lower(),
                    'valid_date': row.get('valid_date', '2024-07-01')
                })
        return questions
    
    else:
        raise ValueError("Dataset must be .json or .csv format")


# =====================================================
# ANSWER EXTRACTION
# =====================================================

def extract_law_info(answer: str, statutes: List[Dict]) -> Dict[str, str]:
    """
    Extract law name, section number, and punishment from generated answer
    and retrieved statutes.
    
    Returns:
        {
            'detected_law': 'IPC' or 'BNS' or 'UNKNOWN',
            'detected_section': '302' or None,
            'detected_punishment': extracted text or None,
            'citation_present': bool
        }
    """
    info = {
        'detected_law': 'UNKNOWN',
        'detected_section': None,
        'detected_punishment': None,
        'citation_present': False
    }
    
    answer_lower = answer.lower()
    
    # Detect law name (BNS takes precedence if both mentioned)
    if 'bns' in answer_lower or 'bharatiya nyaya sanhita' in answer_lower:
        info['detected_law'] = 'BNS'
    elif 'ipc' in answer_lower or 'indian penal code' in answer_lower:
        info['detected_law'] = 'IPC'
    
    # If not in answer, check retrieved statutes
    if info['detected_law'] == 'UNKNOWN' and statutes:
        for statute in statutes:
            title = statute.get('title', '').upper()
            if 'BNS' in title:
                info['detected_law'] = 'BNS'
                break
            elif 'IPC' in title:
                info['detected_law'] = 'IPC'
    
    # Extract section number (e.g., "Section 302", "Sec 103", "§ 378")
    section_patterns = [
        r'section\s+(\d+)',
        r'sec\.?\s+(\d+)',
        r'§\s*(\d+)',
        r'\b(\d{2,3})\s+(?:IPC|BNS)',
    ]
    
    for pattern in section_patterns:
        match = re.search(pattern, answer_lower)
        if match:
            info['detected_section'] = match.group(1)
            info['citation_present'] = True
            break
    
    # Extract punishment keywords
    punishment_keywords = [
        'death', 'life imprisonment', 'rigorous imprisonment',
        'simple imprisonment', 'fine', 'years', 'months'
    ]
    
    for keyword in punishment_keywords:
        if keyword in answer_lower:
            info['detected_punishment'] = keyword
            break
    
    return info


# =====================================================
# METRIC CALCULATIONS
# =====================================================

def calculate_answer_correctness(
    detected_section: str,
    expected_section: str,
    detected_punishment: str,
    expected_punishment: str
) -> bool:
    """
    Check if the answer contains the correct section and punishment.
    
    Answer is correct if:
    - Section number matches exactly, OR
    - Punishment keywords overlap significantly
    """
    section_match = (detected_section == expected_section) if detected_section else False
    
    # Punishment match (keyword overlap)
    punishment_match = False
    if detected_punishment and expected_punishment:
        # Simple keyword containment check
        punishment_match = (
            expected_punishment in detected_punishment or
            detected_punishment in expected_punishment
        )
    
    # Answer is correct if EITHER section or punishment matches
    return section_match or punishment_match


def calculate_temporal_accuracy(
    detected_law: str,
    expected_law: str
) -> bool:
    """
    Temporal Accuracy: Check if system prefers BNS (2024) over IPC.
    
    Returns True if:
    - Expected law is BNS and detected law is BNS ✓
    - Expected law is IPC and detected law is IPC ✓
    
    Returns False if:
    - Expected law is BNS but detected law is IPC ✗ (temporal violation)
    """
    if expected_law == 'BNS':
        return detected_law == 'BNS'
    elif expected_law == 'IPC':
        return detected_law == 'IPC'
    return False


def calculate_law_selection_accuracy(
    detected_law: str,
    expected_law: str
) -> bool:
    """
    Check if the correct law family (IPC or BNS) was selected.
    """
    return detected_law == expected_law


def check_citation_validity(
    detected_law: str,
    detected_section: str,
    statutes: List[Dict]
) -> bool:
    """
    Check if citation is valid:
    - Law name is present (IPC or BNS)
    - Section number is present
    - Section exists in retrieved documents
    """
    if detected_law == 'UNKNOWN' or not detected_section:
        return False
    
    # Check if section appears in any retrieved statute
    for statute in statutes:
        statute_text = statute.get('text', '').lower()
        statute_title = statute.get('title', '').lower()
        
        if detected_section in statute_text or detected_section in statute_title:
            return True
    
    return False


def detect_hallucination(
    detected_law: str,
    detected_section: str,
    detected_punishment: str,
    statutes: List[Dict]
) -> bool:
    """
    Detect hallucination if:
    - Section number doesn't exist in retrieved statutes
    - Law name is completely wrong
    - Punishment is fabricated (not in retrieved text)
    
    Returns True if hallucination detected.
    """
    if not statutes:
        # If no statutes retrieved but answer generated, it's likely hallucinated
        return detected_law != 'UNKNOWN'
    
    # Check if detected section exists in any statute
    section_found = False
    punishment_found = False
    
    for statute in statutes:
        # Include ALL grounded fields: definition (text/content), title, section,
        # and — crucially — the punishment field itself. The previous version only
        # searched 'text' (the definition), so any real punishment was falsely
        # flagged as hallucinated because punishment text never appears in the
        # definition field.
        statute_text = (statute.get('text', '') or statute.get('content', '')).lower()
        statute_title = statute.get('title', '').lower()
        statute_punish = str(statute.get('punishment', '')).lower()
        statute_section = str(statute.get('section', '')).lower()
        grounded = f"{statute_text} {statute_title} {statute_punish} {statute_section}"

        # Check section
        if detected_section and (detected_section.lower() in grounded):
            section_found = True

        # Check punishment against the punishment field (+ other grounded text)
        if detected_punishment and detected_punishment.lower() in grounded:
            punishment_found = True
    
    # Hallucination if section mentioned but not found in sources
    if detected_section and not section_found:
        return True
    
    # Hallucination if specific punishment mentioned but not in sources
    if detected_punishment and not punishment_found:
        return True
    
    return False


# =====================================================
# EVALUATION RUNNER
# =====================================================

def run_evaluation(dataset: List[Dict]) -> Dict[str, float]:
    """
    Run evaluation on the entire test dataset.
    
    Returns metrics as percentages.
    """
    results = {
        'answer_correct': [],
        'temporal_accurate': [],
        'law_selection_correct': [],
        'citation_valid': [],
        'hallucinated': []
    }
    
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80 + "\n")
    
    for idx, test_case in enumerate(dataset):
        question = test_case['question']
        expected_law = test_case['expected_law']
        expected_section = test_case['expected_section']
        expected_punishment = test_case['expected_punishment']
        valid_date = datetime.strptime(test_case['valid_date'], '%Y-%m-%d')
        
        print(f"[{idx+1}/{len(dataset)}] Testing: {question[:60]}...")
        
        # Run through existing RAG pipeline
        try:
            # 1. Retrieve statutes
            statutes = chroma_search(question, valid_date)
            
            # 2. Check if web search needed
            needs_web = any(k in question.lower() for k in ["recent", "latest", "judgment", "court"])
            web_results = serpapi_search(question, valid_date.year) if needs_web else []
            
            # 3. Generate answer
            answer = generate_answer(question, statutes, web_results, valid_date)
            
            # 4. Extract information from answer
            info = extract_law_info(answer, statutes)
            
            # 5. Calculate metrics
            answer_correct = calculate_answer_correctness(
                info['detected_section'],
                expected_section,
                info['detected_punishment'],
                expected_punishment
            )
            
            temporal_accurate = calculate_temporal_accuracy(
                info['detected_law'],
                expected_law
            )
            
            law_correct = calculate_law_selection_accuracy(
                info['detected_law'],
                expected_law
            )
            
            citation_valid = check_citation_validity(
                info['detected_law'],
                info['detected_section'],
                statutes
            )
            
            hallucinated = detect_hallucination(
                info['detected_law'],
                info['detected_section'],
                info['detected_punishment'],
                statutes
            )
            
            # 6. Store results
            results['answer_correct'].append(answer_correct)
            results['temporal_accurate'].append(temporal_accurate)
            results['law_selection_correct'].append(law_correct)
            results['citation_valid'].append(citation_valid)
            results['hallucinated'].append(hallucinated)
            
            # Print individual result
            status = "✓" if answer_correct and temporal_accurate else "✗"
            print(f"    {status} Law: {info['detected_law']} (expected: {expected_law})")
            print(f"       Section: {info['detected_section']} (expected: {expected_section})")
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            # Mark as failures
            results['answer_correct'].append(False)
            results['temporal_accurate'].append(False)
            results['law_selection_correct'].append(False)
            results['citation_valid'].append(False)
            results['hallucinated'].append(True)
    
    # Calculate percentages
    metrics = {
        'Answer Accuracy (%)': (sum(results['answer_correct']) / len(dataset)) * 100,
        'Temporal Accuracy (%)': (sum(results['temporal_accurate']) / len(dataset)) * 100,
        'Law Selection Accuracy (%)': (sum(results['law_selection_correct']) / len(dataset)) * 100,
        'Citation Validity (%)': (sum(results['citation_valid']) / len(dataset)) * 100,
        'Hallucination Rate (%)': (sum(results['hallucinated']) / len(dataset)) * 100
    }
    
    return metrics


# =====================================================
# RESULTS DISPLAY
# =====================================================

def print_evaluation_results(metrics: Dict[str, float]):
    """
    Print evaluation results in a clean table format.
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    
    print("┌─────────────────────────────────┬──────────┐")
    print("│ Metric                          │ Score    │")
    print("├─────────────────────────────────┼──────────┤")
    
    for metric_name, score in metrics.items():
        # Color coding
        if "Hallucination" in metric_name:
            # Lower is better for hallucination
            symbol = "✓" if score < 20 else "✗"
        else:
            # Higher is better for other metrics
            symbol = "✓" if score >= 70 else "✗"
        
        print(f"│ {symbol} {metric_name:<28} │ {score:>6.2f}% │")
    
    print("└─────────────────────────────────┴──────────┘\n")
    
    # Interpretation
    print("INTERPRETATION:")
    print("---------------")
    
    if metrics['Temporal Accuracy (%)'] < 70:
        print("⚠️  Low Temporal Accuracy: System may be preferring IPC over BNS.")
        print("    → Check if ChromaDB contains BNS laws.")
        print("    → Verify date filtering logic.\n")
    
    if metrics['Hallucination Rate (%)'] > 20:
        print("⚠️  High Hallucination Rate: Model generating ungrounded information.")
        print("    → Review prompt engineering in generate_answer().")
        print("    → Check if retrieved statutes are relevant.\n")
    
    if metrics['Citation Validity (%)'] < 70:
        print("⚠️  Low Citation Validity: Model not citing sources properly.")
        print("    → Improve citation prompts.")
        print("    → Verify section extraction logic.\n")
    
    print("="*80 + "\n")


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Legal Assistant Performance')
    parser.add_argument(
        '--dataset',
        type=str,
        default='test_questions.json',
        help='Path to evaluation dataset (JSON or CSV)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        dataset = load_evaluation_dataset(args.dataset)
        print(f"✓ Loaded {len(dataset)} test questions from {args.dataset}")
    except FileNotFoundError:
        print(f"Error: Dataset file '{args.dataset}' not found.")
        print("\nCreating example dataset file...")
        
        # Create example dataset
        example_data = [
            {
                "question": "What is the punishment for murder under Indian law?",
                "expected_law": "BNS",
                "expected_section": "103",
                "expected_punishment": "death or life imprisonment",
                "valid_date": "2024-07-01"
            },
            {
                "question": "What is punishment for theft?",
                "expected_law": "BNS",
                "expected_section": "303",
                "expected_punishment": "imprisonment up to 3 years or fine",
                "valid_date": "2024-07-01"
            },
            {
                "question": "What was the punishment for murder before 2024?",
                "expected_law": "IPC",
                "expected_section": "302",
                "expected_punishment": "death or life imprisonment",
                "valid_date": "2023-01-01"
            }
        ]
        
        with open('test_questions.json', 'w', encoding='utf-8') as f:
            json.dump(example_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created example dataset: test_questions.json")
        print("  Please add more test cases and run again.\n")
        sys.exit(0)
    
    # Run evaluation
    metrics = run_evaluation(dataset)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Save results to file
    results_file = 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}\n")
