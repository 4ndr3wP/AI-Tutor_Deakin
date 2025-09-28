import logging

from _my_tests.measurement_system import OffTopicResult, TestContext


def run_test(context: TestContext) -> OffTopicResult:
    """Tests the system's ability to identify and redirect off-topic questions."""
    logging.info("🎯 Testing RAG-based off-topic detection...")

    off_topic_tests = context.dataset.get('OFF_TOPIC_HANDLING_TESTS', {}).get('cases', [])
    on_topic_tests = context.dataset.get('HALLUCINATION_TESTS', {}).get('test_cases', [])[:3]

    correct_off_topic_detections = 0
    false_positives = 0

    for test in off_topic_tests:
        if 'conversation' in test and test['conversation']:
            query = test['conversation'][0]
            response_data = context.query_system(query, f"off_topic_{test['test_id']}")
        
            if response_data['success']:
                response = response_data['response'].lower()
                redirect_indicators = ['reinforcement learning', 'rl topic', 'course material', 
                                 'unit content', 'back to', 'focus on', 'relating to rl']
            
                if any(indicator in response for indicator in redirect_indicators):
                    correct_off_topic_detections += 1

    for test in on_topic_tests:
        query = test.get('query')
        if query:
            response_data = context.query_system(query, f"on_topic_{test.get('week', 'general')}")
            
            if response_data['success']:
                response = response_data['response'].lower()
                is_redirected = any(phrase in response for phrase in ['back to', 'instead let', 'focus on course'])
                if len(response) < 100 or is_redirected:
                    false_positives += 1
    
    total_tests = len(off_topic_tests) + len(on_topic_tests)
    total_correct = correct_off_topic_detections + (len(on_topic_tests) - false_positives)
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    return OffTopicResult(
        total_tests=total_tests,
        correctly_detected=total_correct,
        false_positives=false_positives,
        false_negatives=len(off_topic_tests) - correct_off_topic_detections,
        detection_accuracy=accuracy,
        avg_retrieval_score=0
    )