import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from _my_tests.measurement_system import SessionIsolationResult, TestContext


def run_test(context: TestContext) -> SessionIsolationResult:
    """
    Tests that concurrent user sessions are properly isolated and do not share information.
    """
    logging.info("🔒 Testing actual session isolation...")
    
    # Scenarios with unique information for each session.
    test_scenarios = [
        {"session": "user_a", "query": "My name is Alice", "follow_up": "What is my name?", "expected": "alice"},
        {"session": "user_b", "query": "My name is Bob", "follow_up": "What is my name?", "expected": "bob"},
        {"session": "user_c", "query": "My favorite topic is Q-learning", "follow_up": "What is my favorite topic?", "expected": "q-learning"}
    ]
    
    # 1. Set up initial context for each session concurrently.
    with ThreadPoolExecutor(max_workers=len(test_scenarios)) as executor:
        setup_futures = [executor.submit(context.query_system, s["query"], s["session"]) for s in test_scenarios]
        for future in as_completed(setup_futures):
            future.result()
    
    time.sleep(1)
    
    # 2. Test isolation by asking follow-up questions concurrently.
    contamination_count = 0
    total_tests = len(test_scenarios)

    with ThreadPoolExecutor(max_workers=total_tests) as executor:
        follow_up_futures = {executor.submit(context.query_system, s["follow_up"], s["session"]): s for s in test_scenarios}
        
        for future in as_completed(follow_up_futures):
            scenario = follow_up_futures[future]
            response_data = future.result()
            
            if not response_data['success']:
                contamination_count += 1
                continue

            response = response_data['response'].lower()
            if scenario['expected'] not in response:
                contamination_count += 1
            
            for other in test_scenarios:
                if other["session"] != scenario["session"] and other['expected'] in response:
                    contamination_count += 1
    
    contamination_count = min(total_tests, contamination_count)
    isolation_score = ((total_tests - contamination_count) / total_tests * 100) if total_tests > 0 else 0
    
    return SessionIsolationResult(
        concurrent_sessions=total_tests,
        memory_bleeding_detected=contamination_count > 0,
        response_contamination=(contamination_count / total_tests * 100) if total_tests > 0 else 0,
        isolation_score=isolation_score
    )