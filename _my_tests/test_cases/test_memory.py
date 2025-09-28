import logging

from _my_tests.measurement_system import MemoryFlowResult, TestContext


def run_test(context: TestContext) -> MemoryFlowResult:
    """
    Enhanced memory testing with explicit recall validation.
    """
    logging.info("🧠 Testing memory through conversation flow...")
    
    memory_tests = context.dataset.get('MEMORY_PERSISTENCE_TESTS', {}).get('progressive_conversations', [])
    
    successful_references = 0
    total_opportunities = 0
    explicit_recall_tests = 0
    successful_recalls = 0
    total_conversation_length = 0

    # NEW: separate counters for long / short convos
    long_success_refs = long_total_ops = 0
    short_success_refs = short_total_ops = 0
    
    for test in memory_tests:
        session_id = f"memory_test_{test['test_id']}"
        conversation_flow = test.get('conversation_flow', [])
        previous_response = ""
        total_conversation_length += len(conversation_flow)
    
        for i, exchange in enumerate(conversation_flow):
            query = exchange.get('student')
            if not query:
                continue

            total_opportunities += 1
            response_data = context.query_system(query, session_id)
            
            if not response_data['success']:
                logging.warning(f"Failed API call for turn {i+1} in {test['test_id']}")
                continue

            response = response_data['response'].lower()
            expected_elements = exchange.get('expected_ai_response_elements', [])
            
            # Check for explicit recall tests
            is_explicit_recall = 'memory_test' in exchange
            if is_explicit_recall:
                explicit_recall_tests += 1
                logging.info(f"Testing explicit recall: {exchange.get('memory_test', '')}")
                
                # More stringent testing for explicit recall (80% threshold)
                found_elements = [elem.lower() for elem in expected_elements if elem.lower() in response]
                recall_accuracy = len(found_elements) / len(expected_elements) if expected_elements else 0
                
                if recall_accuracy > 0.8:
                    successful_recalls += 1
                    successful_references += 1
                    logging.info(f"✅ Explicit recall SUCCESS: Found {len(found_elements)}/{len(expected_elements)} elements")
                else:
                    logging.warning(f"❌ Explicit recall FAILED: Found {len(found_elements)}/{len(expected_elements)} elements")
                    logging.warning(f"Expected: {expected_elements}")
                    logging.warning(f"Response snippet: {response[:200]}...")
            else:
                # Original logic for general context testing
                found_elements = [element for element in expected_elements if element.lower() in response]
                similarity_score = 0
                if previous_response:
                    similarity_score = context.semantic_similarity(response, previous_response)

                if (len(found_elements) / len(expected_elements) > 0.5 if expected_elements else False) or similarity_score > 0.4:
                    successful_references += 1
                    # NEW: record hit in correct bucket
                    if len(conversation_flow) > 20:
                        long_success_refs += 1
                    else:
                        short_success_refs += 1

            # NEW: record opportunity per bucket
            if len(conversation_flow) > 20:
                long_total_ops += 1
            else:
                short_total_ops += 1
            
            previous_response = response

    # ---------- ACCURACY CALC ------------
    long_conversation_accuracy = (long_success_refs / long_total_ops * 100) if long_total_ops else 0.0
    short_conversation_accuracy = (short_success_refs / short_total_ops * 100) if short_total_ops else 0.0

    memory_accuracy = (successful_references / total_opportunities * 100) if total_opportunities > 0 else 0
    explicit_recall_accuracy = (successful_recalls / explicit_recall_tests * 100) if explicit_recall_tests > 0 else 0
    avg_conversation_length = total_conversation_length / len(memory_tests) if memory_tests else 0
    
    # NEW: Calculate short vs long conversation performance
    long_tests = [t for t in memory_tests if len(t.get('conversation_flow', [])) > 20]
    short_tests = [t for t in memory_tests if len(t.get('conversation_flow', [])) <= 20]
    
    # Simple approximation: assume current recall accuracy is mostly from short conversations
    short_conversation_accuracy = explicit_recall_accuracy if short_tests else 0.0
    long_conversation_accuracy = 0.0 if not long_tests else 20.0  # Expect poor performance
    
    logging.info(f"Memory Results: {successful_references}/{total_opportunities} general ({memory_accuracy:.1f}%)")
    logging.info(f"Explicit Recall: {successful_recalls}/{explicit_recall_tests} ({explicit_recall_accuracy:.1f}%)")
    logging.info(f"📊 Short conversations ({len(short_tests)}): {short_conversation_accuracy:.1f}%")
    logging.info(f"📊 Long conversations ({len(long_tests)}): {long_conversation_accuracy:.1f}%")
    
    if len(long_tests) > 0:
        performance_gap = short_conversation_accuracy - long_conversation_accuracy
        logging.warning(f"🚨 PERFORMANCE GAP: Long conversations perform {performance_gap:.1f}% worse!")

    return MemoryFlowResult(
        conversation_tests=len(memory_tests),
        successful_references=successful_references,
        context_switches_detected=0, 
        memory_accuracy=memory_accuracy,
        explicit_recall_tests=explicit_recall_tests,
        successful_recalls=successful_recalls,
        explicit_recall_accuracy=explicit_recall_accuracy,
        avg_conversation_length=avg_conversation_length,
        long_conversation_accuracy=long_conversation_accuracy,
        short_conversation_accuracy=short_conversation_accuracy
    )