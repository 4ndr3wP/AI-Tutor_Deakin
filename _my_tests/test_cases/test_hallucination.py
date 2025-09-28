import logging
import statistics
import os
import re
from typing import Dict, List

from ..measurement_system import HallucinationResult, TestContext


def run_test(context: TestContext) -> HallucinationResult:
    """
    Tests if the system's responses are factually accurate and grounded in course material.
    """
    logging.info("🔍 Testing factual accuracy vs course material...")

    course_content = _load_course_material()
    test_cases = context.dataset.get('HALLUCINATION_TESTS', {}).get('test_cases', [])
    
    factual_count = 0
    similarity_scores = []
    
    for i, test in enumerate(test_cases):
        session_id = f"hallucination_test_{i}"
        response_data = context.query_system(test['query'], session_id, timeout=120)
        
        if response_data['success']:
            response = response_data['response']
            
            is_task = test.get('test_type') == 'task' and 'task_file' in test
            if is_task:
                context_material = _load_specific_task(test['week'], test['task_file'])
                accuracy_score = _check_task_specific_accuracy(context, response, test['expected_topics'], context_material)
            else:
                context_material = course_content.get(test['week'], "")
                accuracy_score = _check_factual_accuracy(context, response, test['expected_topics'], context_material)
            
            similarity_scores.append(accuracy_score)
            
            threshold = 0.3 if is_task else 0.4
            if accuracy_score > threshold:
                factual_count += 1

            logging.info(f"  Test {i+1} ({test['test_type']}): {accuracy_score:.2f} {'✅' if accuracy_score > threshold else '❌'}")
    
    avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0
    factual_accuracy = (factual_count / len(test_cases) * 100) if test_cases else 0
    
    return HallucinationResult(
        total_tests=len(test_cases),
        factual_responses=factual_count,
        hallucinated_responses=len(test_cases) - factual_count,
        avg_similarity_score=avg_similarity,
        factual_accuracy=factual_accuracy
    )

def _load_course_material() -> Dict[str, str]:
    course_content = {}
    weeks_dir = "AI-Tutor/grouped_weeks"
    
    if not os.path.exists(weeks_dir):
        logging.warning(f"Course material directory not found: {weeks_dir}")
        return {}

    for week_folder in os.listdir(weeks_dir):
        week_path = os.path.join(weeks_dir, week_folder)
        if os.path.isdir(week_path):
            content = ""
            for file_name in os.listdir(week_path):
                if file_name.endswith('.md'):
                    file_path = os.path.join(week_path, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content += f.read() + "\n"
                    except Exception as e:
                        logging.warning(f"Could not read file {file_path}: {e}")
            course_content[week_folder] = content
    
    return course_content

def _check_factual_accuracy(context: TestContext, response: str, expected_topics: List[str], course_material: str) -> float:
    if not course_material: return 0.0

    relevant_content = " ".join([
        sentence.strip() for topic in expected_topics for sentence in course_material.split('.')
        if topic.lower() in sentence.lower() and len(sentence.strip()) > 20
    ])

    if not relevant_content: return 0.0

    try:
        return context.semantic_similarity(response, relevant_content)
    except Exception as e:
        logging.warning(f"Similarity calculation failed: {e}")
        response_lower = response.lower()
        matches = sum(1 for topic in expected_topics if topic.lower() in response_lower)
        return matches / len(expected_topics) if expected_topics else 0.0

def _load_specific_task(week: str, task_file: str) -> str:
    task_path = f"AI-Tutor/grouped_weeks/{week}/{task_file}"
    if os.path.exists(task_path):
        try:
            with open(task_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Failed to load task file {task_path}: {e}")
    return ""

def _check_task_specific_accuracy(context: TestContext, response: str, expected_topics: List[str], task_content: str) -> float:
    if not task_content:
        logging.warning("No task content provided for accuracy check.")
        return 0.0
    
    clean_content = re.sub(r'[#*`\-]', '', task_content)
    
    try:
        task_similarity = context.semantic_similarity(response, clean_content)
        
        response_lower = response.lower()
        topic_matches = sum(1 for topic in expected_topics if topic.lower() in response_lower)
        topic_accuracy = topic_matches / len(expected_topics) if expected_topics else 0.0
        
        combined_score = (task_similarity * 0.7) + (topic_accuracy * 0.3)
        
        task_keywords = ['task', 'assignment', 'submit', 'report', 'implement']
        if any(keyword in response_lower for keyword in task_keywords):
            combined_score += 0.1
        
        return min(combined_score, 1.0)
        
    except Exception as e:
        logging.warning(f"Task-specific similarity calculation failed: {e}")
        response_lower = response.lower()
        matches = sum(1 for topic in expected_topics if topic.lower() in response_lower)
        return matches / len(expected_topics) if expected_topics else 0.0