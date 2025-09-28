"""
AI Tutor Measurement System - Redesigned for Meaningful Testing
==============================================================

Focus Areas (per supervisor feedback):
1. RAG-based off-topic detection
2. Cold start vs warm start performance  
3. Memory accuracy through conversation flow
4. Realistic performance thresholds
5. Semantic similarity instead of keyword matching
"""

import json
import time
import statistics
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@dataclass
class TestResult:
    test_id: str
    test_type: str
    success: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: str


@dataclass
class OffTopicResult:
    """Results for RAG-based off-topic detection"""
    total_tests: int
    correctly_detected: int
    false_positives: int
    false_negatives: int
    detection_accuracy: float
    avg_retrieval_score: float


@dataclass
class ColdStartResult:
    """Results comparing cold vs warm start performance"""
    cold_start_time: float
    warm_start_time: float
    performance_degradation: float
    quality_difference: float


@dataclass
class MemoryFlowResult:
    """Results for conversation flow memory testing"""
    conversation_tests: int
    successful_references: int
    context_switches_detected: int
    memory_accuracy: float
    explicit_recall_tests: int = 0
    successful_recalls: int = 0
    explicit_recall_accuracy: float = 0.0
    avg_conversation_length: float = 0.0
    long_conversation_accuracy: float = 0.0  # conversations > 20 turns
    short_conversation_accuracy: float = 0.0  # conversations <= 20 turns

@dataclass
class SessionIsolationResult:
    """Results for actual session isolation testing"""
    concurrent_sessions: int
    memory_bleeding_detected: bool
    response_contamination: float
    isolation_score: float

@dataclass 
class HallucinationResult:
    """Results for hallucination detection testing"""
    total_tests: int
    factual_responses: int
    hallucinated_responses: int
    avg_similarity_score: float
    factual_accuracy: float

@dataclass
class SocraticResult:
    """Results for Socratic response testing"""
    total_tests: int
    socratic_responses: int
    socratic_accuracy: float

@dataclass
class OnTrackTaskResult:
    """Results for OnTrack task retrieval testing"""
    total_tests: int
    successful_retrievals: int
    retrieval_accuracy: float
    task_accuracy: Dict[str, float]
    content_coverage: float


# --- Test Context for Shared Functionality ---

class TestContext:
    """
    Provides shared state and functionality to all individual test cases.
    This object is created once by the TestRunner and passed to each test.
    """
    
    def __init__(self, base_url: str, golden_dataset_path: str):
        self.base_url = base_url
        self.dataset = self._load_golden_dataset(golden_dataset_path)
        self.similarity_model = self._load_similarity_model()
        
        self.thresholds = {
            'avg_response_time': 5.0,
            'memory_accuracy': 70.0,
            'off_topic_detection': 80.0,
            'session_isolation': 95.0,
            'uptime': 90.0,
            'cold_start_degradation': 30.0,
            'hallucination_accuracy': 60.0,
            'socratic_accuracy': 75.0
        }
    
    def _load_similarity_model(self) -> SentenceTransformer:
        logging.info("Loading semantic similarity model...")
        # Use the same Nomic model as the RAG system for consistent semantic evaluation.
        # It's recommended to add `trust_remote_code=True` for this model.
        model_name = "nomic-ai/nomic-embed-text-v1.5"
        try:
            model = SentenceTransformer(model_name, device='cuda:1', trust_remote_code=True)
            logging.info(f"Semantic similarity model '{model_name}' loaded on cuda:1.")
            return model
        except Exception as e:
            logging.warning(f"Could not load model on cuda:1, falling back to CPU. This will be slower. Error: {e}")
            return SentenceTransformer(model_name, trust_remote_code=True)

    def _load_golden_dataset(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Golden dataset not found at: {path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from: {path}")
            raise
    
    def query_system(self, question: str, session_id: str, timeout: int = 90) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": question, "session_id": session_id},
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                return {
                    'success': True, 
                    'response': response.json().get('response', ''),
                    'execution_time': execution_time, 
                    'status_code': 200
                }
            else:
                return {
                    'success': False, 
                    'error': f"HTTP {response.status_code}",
                    'execution_time': execution_time, 
                    'status_code': response.status_code
                }
        except requests.exceptions.Timeout:
            logging.error(f"Request timed out after {timeout} seconds for session {session_id}.")
            return {'success': False, 'error': 'Request timeout', 'execution_time': timeout, 'status_code': 408}
        except Exception as e:
            return {'success': False, 'error': str(e), 'execution_time': time.time() - start_time, 'status_code': 500}

    def semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    