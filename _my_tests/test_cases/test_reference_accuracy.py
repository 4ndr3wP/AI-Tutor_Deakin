"""
Reference Accuracy Test
=======================

Tests that AI Tutor references match actual course materials in grouped_weeks directory.
Also tests that "Related Material" header only appears when actual references are present.
"""

import os
import re
import logging
import uuid
from pathlib import Path
from typing import Dict, Set, List, Tuple
from ..measurement_system import TestContext

class ReferenceResult:
    def __init__(self, total_tests: int, accurate_refs: int, accuracy: float, 
                 display_bug_tests: int = 0, display_bug_failures: int = 0):
        self.total_tests = total_tests
        self.accurate_references = accurate_refs
        self.reference_accuracy = accuracy
        self.display_bug_tests = display_bug_tests
        self.display_bug_failures = display_bug_failures

def run_test(context: TestContext) -> ReferenceResult:
    """Test that references in responses match actual course files and display correctly."""
    logging.info("📚 Testing reference accuracy against course materials...")
    
    # Get actual course structure
    course_files = _get_course_file_structure()
    
    # Test cases that should generate references
    test_questions = [
        "What is Q-learning?",
        "How does SARSA work?", 
        "What is a Markov Decision Process?",
        "Explain temporal difference learning",
        "What are eligibility traces?",
        "How does function approximation work in RL?",
        "What is deep reinforcement learning?",
        "Explain multi-agent reinforcement learning"
    ]
    
    accurate_count = 0
    session_id = f"ref-test-{uuid.uuid4()}"
    
    for i, question in enumerate(test_questions):
        logging.info(f"  - Test {i+1}/{len(test_questions)}: {question}")
        
        result = context.query_system(question, session_id)
        
        if result['success']:
            response = result['response']
            extracted_refs = _extract_references_from_response(response)
            
            # Validate each reference
            valid_refs = _validate_references(extracted_refs, course_files)
            
            if extracted_refs and all(valid_refs.values()):
                accurate_count += 1
                logging.info(f"    ✅ All references valid: {extracted_refs}")
            else:
                invalid_refs = [ref for ref, valid in valid_refs.items() if not valid]
                logging.info(f"    ❌ Invalid references: {invalid_refs}")
        else:
            logging.error(f"    API query failed: {result.get('error')}")
    
    accuracy = (accurate_count / len(test_questions) * 100) if test_questions else 0
    
    logging.info(f"📚 Reference accuracy: {accuracy:.1f}%")
    
    # NEW: Test "Related Material" display bug
    logging.info(" Testing 'Related Material' display behavior...")
    display_bug_results = _test_related_material_display(context)
    
    return ReferenceResult(
        total_tests=len(test_questions),
        accurate_refs=accurate_count,
        accuracy=accuracy,
        display_bug_tests=display_bug_results['total_tests'],
        display_bug_failures=display_bug_results['failures']
    )

def _test_related_material_display(context: TestContext) -> Dict[str, int]:
    """Test that 'Related Material' header only appears when actual references exist."""
    
    # Test cases: in-scope (should have refs) vs out-of-scope (should not have refs)
    test_cases = [
        {
            "question": "What is Q-learning?",
            "expected_refs": True,
            "description": "in-scope question (should have references)"
        },
        {
            "question": "Explain quantum entanglement in photons",
            "expected_refs": False, 
            "description": "out-of-scope question (should not have references)"
        },
        {
            "question": "How do neural networks work in deep learning?",
            "expected_refs": True,
            "description": "in-scope question (should have references)"
        },
        {
            "question": "What is the capital of Mars?",
            "expected_refs": False,
            "description": "out-of-scope question (should not have references)"
        },
        {
            "question": "Explain the theory of relativity",
            "expected_refs": False,
            "description": "out-of-scope question (should not have references)"
        }
    ]
    
    failures = 0
    session_id = f"display-test-{uuid.uuid4()}"
    
    for i, test_case in enumerate(test_cases):
        logging.info(f"  - Display Test {i+1}/{len(test_cases)}: {test_case['description']}")
        
        result = context.query_system(test_case['question'], session_id)
        
        if result['success']:
            response = result['response']
            has_refs = _has_actual_references(response)
            has_header = _has_related_material_header(response)
            
            # Check if display behavior is correct
            if test_case['expected_refs']:
                # Should have both references AND header
                if has_refs and has_header:
                    logging.info(f"    ✅ Correct: Has refs and header")
                elif has_refs and not has_header:
                    logging.warning(f"    ⚠️  Missing header with valid references")
                    failures += 1
                elif not has_refs and has_header:
                    logging.error(f"    ❌ BUG: Has header but no references!")
                    failures += 1
                else:
                    logging.warning(f"    ⚠️  No references found (may be expected)")
            else:
                # Should have neither references NOR header
                if not has_refs and not has_header:
                    logging.info(f"    ✅ Correct: No refs, no header")
                elif not has_refs and has_header:
                    logging.error(f"    ❌ BUG: Has header but no references!")
                    failures += 1
                elif has_refs and has_header:
                    logging.warning(f"    ⚠️  Unexpected references found")
                else:
                    logging.info(f"    ✅ Correct: No refs, no header")
        else:
            logging.error(f"    API query failed: {result.get('error')}")
            failures += 1
    
    bug_rate = (failures / len(test_cases) * 100) if test_cases else 0
    logging.info(f"🔍 Display bug rate: {bug_rate:.1f}% ({failures}/{len(test_cases)} failures)")
    
    return {
        'total_tests': len(test_cases),
        'failures': failures
    }

def _has_actual_references(response: str) -> bool:
    """Check if response contains actual course material references."""
    # Look for patterns like [Week X slides], [Week X task], etc.
    ref_patterns = [
        r'\[Week\s+\d+\s+(?:slides?|tasks?)\]',  # [Week 5 slides]
        r'\[Week\s+\d+\s*\([^)]+\)\]',           # [Week 5 (SIT796-5.1P)]
        r'Week\s+\d+(?:\s*,\s*Slide\s+\d+)?',    # Week 5, Slide 3
    ]
    
    for pattern in ref_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False

def _has_related_material_header(response: str) -> bool:
    """Check if response contains 'Related Material' header."""
    # Look for "Related material:" or "Related Material:" patterns
    header_patterns = [
        r'Related\s+material\s*:',
        r'Related\s+Material\s*:',
    ]
    
    for pattern in header_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False

def _get_course_file_structure() -> Dict[str, Set[str]]:
    """Get actual course file structure from grouped_weeks directory."""
    course_files = {}
    
    # Fix the path - when running from _my_tests directory, need to go up one level
    weeks_dir = Path("../AI-Tutor/grouped_weeks")
    if not weeks_dir.exists():
        logging.warning(f"Course directory not found: {weeks_dir}")
        return {}
    
    logging.info(f"✅ Found course directory: {weeks_dir}")
    
    for week_dir in sorted(weeks_dir.iterdir()):
        if week_dir.is_dir() and week_dir.name.startswith('week'):
            week_num = week_dir.name.replace('week', '').zfill(2)
            course_files[week_num] = set()
            
            for file_path in week_dir.iterdir():
                if file_path.suffix == '.md':
                    if file_path.name.startswith('SIT796-'):
                        course_files[week_num].add('task')
                    elif file_path.name.startswith('Week'):
                        course_files[week_num].add('slides')
    
    logging.info(f"📁 Course structure loaded: {course_files}")
    return course_files

def _extract_references_from_response(response: str) -> List[str]:
    """Extract references from response text with flexible patterns."""
    references = []
    
    # Pattern 1: [Week X slides], [Week X task] (original format)
    pattern1 = r'\[([Ww]eek\s+\d+\s+(?:slides?|tasks?))\]'
    matches1 = re.findall(pattern1, response, re.IGNORECASE)
    references.extend(matches1)
    
    # Pattern 2: Week X, Slide Y (common format)
    pattern2 = r'[Ww]eek\s+(\d+)(?:\s*,\s*[Ss]lide\s+\d+)?'
    matches2 = re.findall(pattern2, response, re.IGNORECASE)
    for week in matches2:
        references.append(f"Week {week} slides")
    
    # Pattern 3: [Weeks X, Y, Z] (range format)
    pattern3 = r'\[[Ww]eeks?\s+([\d,\s]+)\]'
    matches3 = re.findall(pattern3, response, re.IGNORECASE)
    for week_list in matches3:
        weeks = re.findall(r'\d+', week_list)
        for week in weeks:
            references.append(f"Week {week} slides")
    
    # Pattern 4: Related material: [Week X] (simple format)
    pattern4 = r'[Rr]elated\s+[Mm]aterial:?\s*\[?[Ww]eek\s+(\d+)\]?'
    matches4 = re.findall(pattern4, response, re.IGNORECASE)
    for week in matches4:
        references.append(f"Week {week} slides")
    
    # Remove duplicates and return
    return list(set(references))

def _validate_references(references: List[str], course_files: Dict[str, Set[str]]) -> Dict[str, bool]:
    """Validate that references match actual course files."""
    validation = {}
    
    for ref in references:
        # Parse reference: "Week 05 slides" -> week="05", type="slides"
        match = re.match(r'week\s+(\d+)\s+(slides?|tasks?)', ref, re.IGNORECASE)
        if match:
            week_num = match.group(1).zfill(2)
            ref_type = 'slides' if 'slide' in match.group(2).lower() else 'task'
            
            # Check if this week and type exist in course files
            exists = week_num in course_files and ref_type in course_files[week_num]
            validation[ref] = exists
        else:
            validation[ref] = False
    
    return validation