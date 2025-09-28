import logging
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from _my_tests.measurement_system import TestContext


def run_test(context: TestContext) -> Dict[str, Any]:
    """
    Tests system performance under a simulated concurrent load.
    """
    logging.info("⚡ Testing system performance with realistic thresholds...")
    
    test_queries = context.dataset.get('EDGE_CASE_TESTS', {}).get('rapid_fire_tests', [{}])[0].get('questions', [])
    
    response_times = []
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_query = {
            executor.submit(context.query_system, query, f"perf_test_{i}", timeout=30): query 
            for i, query in enumerate(test_queries)
        }
        for future in as_completed(future_to_query):
            result = future.result()
            if result['success']:
                response_times.append(result['execution_time'])
                success_count += 1
    
    if response_times:
        avg_time = statistics.mean(response_times)
        p90_time = np.percentile(response_times, 90)
        p99_time = np.percentile(response_times, 99)
        max_time = max(response_times)
        min_time = min(response_times)
    else:
        avg_time = p90_time = p99_time = max_time = min_time = float('inf')
    
    success_rate = (success_count / len(test_queries) * 100) if test_queries else 0

    return {
        'total_requests': len(test_queries),
        'successful_requests': success_count,
        'success_rate': success_rate,
        'average_response_time': avg_time,
        'p90_latency': p90_time,
        'p99_latency': p99_time,
        'max_response_time': max_time,
        'min_response_time': min_time
    }