"""
OnTrack Task Retrieval Test
===========================

Tests that AI Tutor can correctly retrieve and provide information about 
specific OnTrack tasks from the grouped_weeks directory.

This test covers all 17 OnTrack tasks:
Pass Tasks (P): 1.1P, 2.1P, 3.1P, 4.1P, 5.1P, 6.1P, 7.1P, 8.1P, 9.1P, 10.1P
Credit Tasks (C): 3.2C, 6.2C, 9.2C  
Distinction Tasks (D): 4.2D, 10.2D
High Distinction Tasks (HD): 5.2HD, 8.2HD
"""

import logging
import uuid
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from ..measurement_system import TestContext

class OnTrackTaskResult:
    """Results for OnTrack task retrieval testing"""
    def __init__(self, total_tests: int, successful_retrievals: int, 
                 task_accuracy: Dict[str, float], content_coverage: float):
        self.total_tests = total_tests
        self.successful_retrievals = successful_retrievals
        self.retrieval_accuracy = (successful_retrievals / total_tests * 100) if total_tests > 0 else 0
        self.task_accuracy = task_accuracy
        self.content_coverage = content_coverage

def run_test(context: TestContext) -> OnTrackTaskResult:
    """
    Test that AI Tutor can retrieve specific OnTrack task information
    """
    logging.info("📋 Testing OnTrack task retrieval accuracy...")
    
    # Load actual OnTrack task content for validation
    task_files = _load_ontrack_tasks()
    
    # Test cases for all OnTrack tasks
    test_cases = _get_ontrack_test_cases()
    
    successful_retrievals = 0
    task_accuracy = {}
    content_scores = []
    
    for test_case in test_cases:
        session_id = f"ontrack_test_{uuid.uuid4().hex[:8]}"
        task_id = test_case['task_id']
        
        # Query the AI Tutor about the specific task
        response_data = context.query_system(test_case['query'], session_id, timeout=120)
        
        if response_data['success']:
            response = response_data['response'].lower()
            
            # Check if response contains expected task-specific content
            expected_keywords = test_case['expected_keywords']
            found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response)
            
            task_score = (found_keywords / len(expected_keywords)) * 100 if expected_keywords else 0
            task_accuracy[task_id] = task_score
            
            # Check if response references the correct task file
            file_referenced = any(ref in response for ref in [task_id.lower(), f"task {task_id.split('-')[1]}"])
            
            if task_score >= 50 and file_referenced:  # At least 50% content match + task reference
                successful_retrievals += 1
            
            content_scores.append(task_score)
            
            logging.info(f"Task {task_id}: {task_score:.1f}% content match, Referenced: {file_referenced}")
        else:
            task_accuracy[task_id] = 0
            content_scores.append(0)
            logging.warning(f"Failed to query task {task_id}")
    
    overall_content_coverage = sum(content_scores) / len(content_scores) if content_scores else 0
    
    logging.info(f"OnTrack Results: {successful_retrievals}/{len(test_cases)} tasks successfully retrieved")
    logging.info(f"Overall content coverage: {overall_content_coverage:.1f}%")
    
    return OnTrackTaskResult(
        total_tests=len(test_cases),
        successful_retrievals=successful_retrievals,
        task_accuracy=task_accuracy,
        content_coverage=overall_content_coverage
    )

def _load_ontrack_tasks() -> Dict[str, str]:
    """Load all OnTrack task files for content validation"""
    task_files = {}
    grouped_weeks_path = Path(__file__).parent.parent.parent / "AI-Tutor" / "grouped_weeks"
    
    for week_dir in grouped_weeks_path.glob("week*"):
        for task_file in week_dir.glob("SIT796-*.md"):
            task_id = task_file.stem  # e.g., "SIT796-1.1P"
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_files[task_id] = f.read()
            except Exception as e:
                logging.warning(f"Could not load {task_file}: {e}")
    
    return task_files

def _get_ontrack_test_cases() -> List[Dict]:
    """Generate comprehensive test cases for all OnTrack tasks"""
    return [
        # Week 1 - Pass Task 1.1P
        {
            'task_id': 'SIT796-1.1P',
            'query': 'What is Task 1.1P about and what should I include?',
            'expected_keywords': ['gym environment', 'states', 'actions', 'transition functions', 'rewards', 
                                'discrete', 'continuous', 'deterministic', 'probabilistic', 'optimal policy', '600 words']
        },
        {
            'task_id': 'SIT796-1.1P',
            'query': 'Tell me about the requirements for OnTrack Task 1.1P',
            'expected_keywords': ['gymnasium.farama.org', 'case study', 'environments', 'Classic Control', 'Box2D']
        },
        
        # Week 2 - Pass Task 2.1P  
        {
            'task_id': 'SIT796-2.1P',
            'query': 'What does Task 2.1P ask about the AI-powered cleaning robot?',
            'expected_keywords': ['cleaning robot', 'supervised learning', 'unsupervised learning', 
                                'reinforcement learning', 'features', 'value computation', 'discount factor', '500 words']
        },
        {
            'task_id': 'SIT796-2.1P',
            'query': 'Help me understand the treasure state problem in Task 2.1P',
            'expected_keywords': ['treasure state', 'gamma = 0.9', 'rr = 1', 'rother = -0.1', 'discounted', 'undiscounted']
        },
        
        # Week 3 - Pass Task 3.1P
        {
            'task_id': 'SIT796-3.1P',
            'query': 'What are the main questions in Task 3.1P about dynamic programming?',
            'expected_keywords': ['policy evaluation', 'policy improvement', 'policy iteration', 
                                'limitations', 'dynamic programming', '500-600 words']
        },
        
        # Week 3 - Credit Task 3.2C
        {
            'task_id': 'SIT796-3.2C',
            'query': 'What should I implement for the epsilon strategy in Task 3.2C?',
            'expected_keywords': ['linearly decaying epsilon', 'epsilon=1', 'reduce by 0.005', 'minimum 0.01',
                                'MiniGrid-Empty-6x6-v0', 'fixed epsilon', '0.1', '0.9', 'reward curves', 'plot']
        },
        {
            'task_id': 'SIT796-3.2C',
            'query': 'Tell me about the cleaning robot controller question in Task 3.2C',
            'expected_keywords': ['automated cleaning robot', 'dynamic programming', 'controller', 'aspects', 'algorithm']
        },
        
        # Week 4 - Pass Task 4.1P
        {
            'task_id': 'SIT796-4.1P',
            'query': 'What does Task 4.1P ask about on-policy and off-policy?',
            'expected_keywords': ['on-policy', 'off-policy', 'algorithms', 'gamma set to 1', 'workshop', '300-500 words']
        },
        
        # Week 4 - Distinction Task 4.2D
        {
            'task_id': 'SIT796-4.2D',
            'query': 'What does Task 4.2D ask about the robotic arm controller?',
            'expected_keywords': ['robotic arm', 'picking up objects', 'Dynamic programming', 'Monte-Carlo methods',
                                'benefits', 'drawbacks', 'faulty robot', 'joints', 'navigation environment']
        },
        {
            'task_id': 'SIT796-4.2D',
            'query': 'Tell me about the 100x100 environment challenge in Task 4.2D',
            'expected_keywords': ['100x100 environment', 'challenging', 'implementation easier', 'Monte-Carlo Control']
        },
        
        # Week 5 - Pass Task 5.1P
        {
            'task_id': 'SIT796-5.1P',
            'query': 'What does Task 5.1P ask about Q-learning and SARSA?',
            'expected_keywords': ['Cliff Walking', 'SARSA', 'Q-learning', 'safer policies', 'fixed epsilon', 'decaying epsilon']
        },
        
        # Week 5 - High Distinction Task 5.2HD
        {
            'task_id': 'SIT796-5.2HD',
            'query': 'What is the robot navigation problem in Task 5.2HD?',
            'expected_keywords': ['robot', 'room', 'green lines', 'trajectory', 'goal location', 'g1', 'g2', 'g3',
                                'gray area', 'policy', 'algorithm']
        },
        {
            'task_id': 'SIT796-5.2HD',
            'query': 'Tell me about the TD errors plot in Task 5.2HD',
            'expected_keywords': ['epsilon=1', 'random actions', 'TD errors', 'absolute values', 'episodes', 'SARSA', 'Q-learning']
        },
        
        # Week 6 - Pass Task 6.1P
        {
            'task_id': 'SIT796-6.1P',
            'query': 'What does Task 6.1P ask about Eligibility Traces?',
            'expected_keywords': ['Eligibility Traces', 'main advantage', 'DYNA', 'advantage', 'disadvantage']
        },
        
        # Week 6 - Credit Task 6.2C
        {
            'task_id': 'SIT796-6.2C',
            'query': 'What does Task 6.2C ask about accumulating traces?',
            'expected_keywords': ['accumulating traces', 'associate', 'function', 'ChatGPT', 'GenAl tool', 'code correct']
        },
        {
            'task_id': 'SIT796-6.2C',
            'query': 'Tell me about the Dyna-Q pseudocode question in Task 6.2C',
            'expected_keywords': ['Dyna-Q', 'pseudocode', 'highlighted line', 'SARSA update', 'DYNASARSA']
        },
        
        # Week 7 - Pass Task 7.1P
        {
            'task_id': 'SIT796-7.1P',
            'query': 'What does Task 7.1P ask about function approximation?',
            'expected_keywords': ['deadly triad', 'reinforcement learning', 'tile coding', 'principle', 'tiles', 'tilings']
        },
        
        # Week 8 - Pass Task 8.1P
        {
            'task_id': 'SIT796-8.1P',
            'query': 'What are the main questions in Task 8.1P about Deep RL?',
            'expected_keywords': ['tabular Q-learning', 'DQN', 'tricks', 'policy gradient', 'value-based methods']
        },
        
        # Week 8 - High Distinction Task 8.2HD
        {
            'task_id': 'SIT796-8.2HD',
            'query': 'What does Task 8.2HD ask about implementing double DQN?',
            'expected_keywords': ['double DQN', 'CartPole', 'learning curve', '5 seeds', 'learning steps', 
                                'standard error', 'shaded regions', 'environment interactions']
        },
        {
            'task_id': 'SIT796-8.2HD',
            'query': 'Tell me about the Google Colab credits mention in Task 8.2HD',
            'expected_keywords': ['$100', 'google colab credits', 'standard environment', 'exhaust']
        },
        
        # Week 9 - Pass Task 9.1P
        {
            'task_id': 'SIT796-9.1P',
            'query': 'What does Task 9.1P ask about multiagent reinforcement learning?',
            'expected_keywords': ['multiagent reinforcement learning', 'main challenges', 'action-advising strategies', 'lecture']
        },
        
        # Week 9 - Credit Task 9.2C
        {
            'task_id': 'SIT796-9.2C',
            'query': 'What does Task 9.2C ask about centralised vs decentralised execution?',
            'expected_keywords': ['centralised training', 'decentralised execution', 'MARL', 'trade-offs', 
                                'cooperative game', 'shared reward', 'credit assignment']
        },
        
        # Week 10 - Pass Task 10.1P
        {
            'task_id': 'SIT796-10.1P',
            'query': 'What does Task 10.1P ask about Multi-Objective Reinforcement Learning?',
            'expected_keywords': ['multiobjective reinforcement learning', 'single-objective', 'mo-gymnasium.farama.org',
                                'grid-world', 'cleaning robot', 'MORL', 'safety', 'sustainability']
        },
        
        # Week 10 - Distinction Task 10.2D
        {
            'task_id': 'SIT796-10.2D',
            'query': 'What does Task 10.2D ask about literature review?',
            'expected_keywords': ['survey paper', 'summarise literature', 'sub-topics', 'related', 'illustrative',
                                'creative', 'figures', 'video', '1500 words', 'OneDrive folder']
        },
        {
            'task_id': 'SIT796-10.2D',
            'query': 'Tell me about the research areas listed in Task 10.2D',
            'expected_keywords': ['Explainable RL', 'Safe RL', 'Continual RL', 'MultiAgent RL', 'MultiObjective RL',
                                'Curriculum Learning', 'Human-in-the-loop', 'Model-based RL', 'Meta-RL']
        }
    ]