import logging, random, csv, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from _my_tests.measurement_system import TestContext

def run_test(context: TestContext):
    logging.info("🚦 Starting GPU-stress (concurrent users) test...")
    
    # Test different concurrency levels to find breaking point
    concurrency_levels = [2, 5, 10, 15, 20]  # Graduated load testing
    sessions = [f"stress_{i}" for i in range(max(concurrency_levels))]

    # Pick any progressive conversations from golden dataset
    conv_pool = context.dataset["MEMORY_PERSISTENCE_TESTS"]["progressive_conversations"]
    samples = random.choices(conv_pool, k=len(sessions))

    def simulate_realistic_user(user_id, num_requests=3):
        """Simulate a realistic user with thinking time and varied behavior"""
        session_id = f"user_{user_id}"
        successful_requests = 0
        failed_requests = 0
        
        # Each user gets a random conversation
        convo = random.choice(samples)["conversation_flow"][:num_requests]
        
        for i, turn in enumerate(convo):
            q = turn["student"]
            
            # Realistic thinking time between requests (varies by user)
            if i > 0:  # Don't delay first request
                thinking_time = random.uniform(1.0, 5.0)  # 1-5 seconds
                logging.info(f"User {user_id} thinking for {thinking_time:.1f}s...")
                time.sleep(thinking_time)
            
            # Make request with timeout
            result = context.query_system(q, session_id, timeout=120)
            
            if result['success']:
                successful_requests += 1
                logging.info(f"✅ User {user_id} request {i+1} succeeded")
            else:
                failed_requests += 1
                logging.error(f"❌ User {user_id} request {i+1} failed: {result.get('error', 'Unknown error')}")
                # If user fails, they might retry or give up
                if random.random() < 0.3:  # 30% chance of retry
                    logging.info(f"User {user_id} retrying...")
                    time.sleep(random.uniform(0.5, 2.0))
                    retry_result = context.query_system(q, session_id, timeout=60)
                    if retry_result['success']:
                        successful_requests += 1
                        failed_requests -= 1
                        logging.info(f"✅ User {user_id} retry succeeded")
        
        return {
            'user_id': user_id,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_requests': num_requests
        }

    # Test each concurrency level
    results = {}
    
    for concurrency in concurrency_levels:
        logging.info(f"🧪 Testing {concurrency} concurrent users...")
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        total_requests = 0
        
        # Run concurrent users with realistic behavior
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all user tasks
            futures = [executor.submit(simulate_realistic_user, i) for i in range(concurrency)]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    user_result = future.result()
                    successful_requests += user_result['successful_requests']
                    failed_requests += user_result['failed_requests']
                    total_requests += user_result['total_requests']
                except Exception as e:
                    logging.error(f"User simulation failed: {e}")
                    failed_requests += 3  # Assume 3 failed requests per failed user
        
        duration = time.time() - start_time
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        results[concurrency] = {
            'success_rate': success_rate,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'total_requests': total_requests,
            'duration': duration
        }
        
        logging.info(f"📊 {concurrency} users: {success_rate:.1f}% success rate, {duration:.1f}s")
        
        # If success rate drops below 80%, stop testing higher concurrency
        if success_rate < 80.0:
            logging.warning(f" System degraded at {concurrency} users (success rate: {success_rate:.1f}%)")
            break

    # Parse GPU CSV written by TestRunner
    gpu_util, gpu_mem = [], []
    try:
        with open(context.gpu_log_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gpu_util.append(float(row[' utilization.gpu [%]'].rstrip('%')))
                gpu_mem.append(float(row[' memory.used [MiB]'].rstrip(' MiB')))
        peak_util = max(gpu_util) if gpu_util else 0
        peak_mem  = max(gpu_mem)  if gpu_mem  else 0
    except Exception as e:
        logging.warning(f"Could not parse GPU log: {e}")
        peak_util = peak_mem = -1

    # Find the maximum stable concurrency
    max_stable_concurrency = max([k for k, v in results.items() if v['success_rate'] >= 80.0], default=0)
    overall_success_rate = results[max_stable_concurrency]['success_rate'] if max_stable_concurrency > 0 else 0
    
    return {
        "peak_gpu_util": peak_util,
        "peak_gpu_mem_mib": peak_mem,
        "max_stable_concurrency": max_stable_concurrency,
        "overall_success_rate": overall_success_rate,
        "concurrency_results": results,
        "sessions": max(concurrency_levels)
    }
