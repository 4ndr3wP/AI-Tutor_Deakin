"""
AI Tutor Socratic Response Test - Enhanced
==========================================

Tests the AI Tutor's ability to respond Socratically, including:
1. Basic Socratic questioning behavior
2. Code generation prevention
3. Long prompt rejection
4. Educational guidance over direct answers
"""

import uuid
import logging
from ..measurement_system import TestContext, SocraticResult

def run_test(context: TestContext) -> SocraticResult:
    """
    Executes comprehensive Socratic response testing.
    """
    logging.info("🧠 Starting comprehensive Socratic response test...")
    
    # Test 1: Basic Socratic behavior from dataset
    basic_socratic_tests = context.dataset.get('SOCRATIC_TUTOR_TESTS', {}).get('requests', [])
    
    # Test 2: Code generation prevention tests
    code_prevention_tests = [
        "Write the complete Python code for Q-learning algorithm",
        "Give me the full implementation of SARSA with epsilon decay", 
        "Show me all the code for the MiniGrid environment setup",
        "Can you write the entire DQN implementation?",
        "def q_learning(): # I need help with this function",
        "import numpy as np # Help me with this import"
    ]
    
    # Test 3: Long prompt rejection tests
    long_prompt_tests = [
        "What is reinforcement learning? " * 100,  # ~3000 characters
        "Explain Q-learning " + "in detail " * 200,  # ~2000 characters
    ]
    
    # Test 4: Direct answer suppression tests
    direct_answer_tests = [
        "What is the exact formula for the Bellman equation?",
        "Tell me the precise steps of the Q-learning algorithm",
        "Give me the answer to task 2.1P",
        "What are the exact hyperparameters for DQN?",
        "Just tell me the solution to the cliff walking problem"
    ]
    
    all_tests = []
    test_categories = []
    
    # Combine all test cases with category labels
    for test in basic_socratic_tests:
        all_tests.append(test["prompt"])
        test_categories.append("basic_socratic")
    
    for test in code_prevention_tests:
        all_tests.append(test)
        test_categories.append("code_prevention")
    
    for test in long_prompt_tests:
        all_tests.append(test)
        test_categories.append("long_prompt")
    
    for test in direct_answer_tests:
        all_tests.append(test)
        test_categories.append("direct_answer_suppression")
    
    if not all_tests:
        logging.warning("No test cases found. Skipping Socratic test.")
        return SocraticResult(total_tests=0, socratic_responses=0, socratic_accuracy=0.0)

    socratic_responses = 0
    code_prevention_success = 0
    long_prompt_rejections = 0
    direct_answer_suppressions = 0
    session_id = f"socratic-test-{uuid.uuid4()}"

    for i, (test_case, category) in enumerate(zip(all_tests, test_categories)):
        logging.info(f"  - Test {i+1}/{len(all_tests)} ({category}): {test_case[:50]}...")
        
        result = context.query_system(test_case, session_id)
        
        if result['success']:
            response_text = result['response'].strip()
            
            # Category-specific evaluation
            if category == "basic_socratic":
                if response_text.endswith('?'):
                    socratic_responses += 1
                    logging.info(f"    ✅ Socratic response detected.")
                else:
                    logging.info(f"    ❌ Non-Socratic response: {response_text[:100]}...")
            
            elif category == "code_prevention":
                is_question = response_text.endswith('?')
                has_code = any(indicator in response_text.lower() for indicator in 
                             ['def ', 'class ', 'import ', '```', 'for i in', 'while '])
                
                if is_question and not has_code:
                    code_prevention_success += 1
                    socratic_responses += 1
                    logging.info(f"    ✅ Code prevention successful (question without code)")
                else:
                    logging.info(f"    ❌ Code generated or non-question response")
            
            elif category == "long_prompt":
                # Check if response indicates rejection or educational message
                educational_indicators = ['turn-based', 'mass posting', 'smaller', 'focused parts']
                if any(indicator in response_text.lower() for indicator in educational_indicators):
                    long_prompt_rejections += 1
                    logging.info(f"    ✅ Long prompt properly rejected with educational message")
                else:
                    logging.info(f"    ❌ Long prompt not properly rejected")
            
            elif category == "direct_answer_suppression":
                is_question = response_text.endswith('?')
                has_direct_answer = any(indicator in response_text.lower() for indicator in 
                                      ['the answer is', 'the solution is', 'here is the', 'the formula is'])
                
                if is_question and not has_direct_answer:
                    direct_answer_suppressions += 1
                    socratic_responses += 1
                    logging.info(f"    ✅ Direct answer suppressed, question provided")
                else:
                    logging.info(f"    ❌ Direct answer provided instead of question")
        
        else:
            # For long prompts, API failure might be expected (rejection)
            if category == "long_prompt":
                long_prompt_rejections += 1
                logging.info(f"    ✅ Long prompt rejected (API error)")
            else:
                logging.error(f"    API query failed: {result.get('error')}")

    # Calculate overall accuracy
    total_tests = len(all_tests)
    overall_accuracy = (socratic_responses / total_tests * 100) if total_tests > 0 else 0
    
    # Log detailed results
    logging.info(f"🧠 Comprehensive Socratic test results:")
    logging.info(f"  📊 Overall Socratic accuracy: {overall_accuracy:.1f}%")
    logging.info(f"   Code prevention: {code_prevention_success}/{len(code_prevention_tests)}")
    logging.info(f"   Long prompt rejections: {long_prompt_rejections}/{len(long_prompt_tests)}")
    logging.info(f"   Direct answer suppressions: {direct_answer_suppressions}/{len(direct_answer_tests)}")
    
    return SocraticResult(
        total_tests=total_tests,
        socratic_responses=socratic_responses,
        socratic_accuracy=overall_accuracy
    )