import logging
import time

from _my_tests.measurement_system import ColdStartResult, TestContext


def run_test(context: TestContext) -> ColdStartResult:
    """
    Measures the performance difference between a cold start and a warm start query.
    """
    logging.info("🌡️ Testing cold start vs warm start performance...")
    
    test_query = "What is reinforcement learning?"
    session_id = f"cold_warm_test_{int(time.time())}"
    
    cold_result = context.query_system(test_query, session_id)
    cold_time = cold_result.get('execution_time', 0)
    cold_response = cold_result.get('response', '')
    
    time.sleep(2)
    
    warm_result = context.query_system(test_query, session_id)
    warm_time = warm_result.get('execution_time', 0)
    warm_response = warm_result.get('response', '')
    
    quality_diff = context.semantic_similarity(cold_response, warm_response)
    degradation = ((cold_time - warm_time) / warm_time * 100) if warm_time > 0 else 0
    
    return ColdStartResult(
        cold_start_time=cold_time,
        warm_start_time=warm_time,
        performance_degradation=degradation,
        quality_difference=quality_diff
    )