"""
AI Tutor Test Runner
========================================================

A command-line tool for executing a comprehensive test suite against the AI Tutor API.
This runner can execute all tests to generate a full report or run specific tests individually.
"""

import argparse
import json
import time
import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Add project root to sys.path ---
# This allows absolute imports from the project root and makes the script runnable directly.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from _my_tests.measurement_system import TestContext
# Import the individual test case functions
from _my_tests.test_cases import (
    test_cold_start,
    test_hallucination,
    test_isolation,
    test_memory,
    test_off_topic,
    test_performance,
    test_socratic,
    test_gpu_stress,
    test_reference_accuracy,
    test_ontrack_tasks
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class TestRunner:
    """Orchestrates the execution of the AI Tutor test suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.context = TestContext(base_url, "golden_dataset.json")
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.gpu_monitor_proc: Optional[subprocess.Popen] = None
        self.gpu_log_path: Optional[str] = None
            
    def start_gpu_monitoring(self):
        """Starts a background process to monitor GPU usage using nvidia-smi."""
        try:
            results_dir = Path("results")
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.gpu_log_path = str(results_dir / f"gpu_{ts}.csv")
            self.context.gpu_log_path = self.gpu_log_path
            
            command = ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv", "-l", "5"]

            with open(self.gpu_log_path, "w") as f:
                self.gpu_monitor_proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
            logging.info(f"🖥️ GPU monitoring started. Log file: {self.gpu_log_path}")
            
        except FileNotFoundError:
            logging.warning("`nvidia-smi` command not found. GPU monitoring will be skipped.")
        except Exception as e:
            logging.warning(f"Could not start GPU monitoring: {e}")
    
    def stop_gpu_monitoring(self):
        """Stops the GPU monitoring background process gracefully."""
        if not self.gpu_monitor_proc:
            return
            
        try:
            self.gpu_monitor_proc.terminate()
            self.gpu_monitor_proc.wait(timeout=5)
            logging.info("🖥️ GPU monitoring stopped.")
        except subprocess.TimeoutExpired:
            logging.warning("GPU monitor did not terminate gracefully. Forcing kill.")
            try:
                self.gpu_monitor_proc.kill()
            except Exception as e:
                logging.error(f"Failed to kill GPU monitor process: {e}")
        finally:
            self.gpu_monitor_proc = None
    
    def check_system_health(self) -> Dict[str, Any]:
        """Performs a quick health check on the AI Tutor API."""
        logging.info("🏥 Checking system health...")
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logging.info(f"✅ System is healthy. Model: {health_data.get('model')}")
                return {'healthy': True, 'model': health_data.get('model')}
            else:
                logging.error(f"Health check failed with HTTP status {response.status_code}.")
                return {'healthy': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            logging.error(f"Health check failed with exception: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def _run_and_process_test(self, test_name: str, test_function: callable) -> Dict[str, Any]:
        """A generic wrapper to run a test function and process its results."""
        result = test_function(self.context)
        thresholds = self.context.thresholds
        
        # Determine success based on the test's key metric
        success = False
        key_metric_value = None
        if test_name == 'off-topic':
            key_metric_value = result.detection_accuracy
            success = key_metric_value >= thresholds['off_topic_detection']
        elif test_name == 'cold-start':
            key_metric_value = result.performance_degradation
            success = key_metric_value <= thresholds['cold_start_degradation']
        elif test_name == 'memory':
            key_metric_value = result.memory_accuracy
            success = key_metric_value >= thresholds['memory_accuracy']
        elif test_name == 'isolation':
            key_metric_value = result.isolation_score
            success = key_metric_value >= thresholds['session_isolation']
        elif test_name == 'performance':
            key_metric_value = result['average_response_time']
            time_ok = key_metric_value <= thresholds['avg_response_time']
            uptime_ok = result['success_rate'] >= thresholds['uptime']
            success = time_ok and uptime_ok
        elif test_name == 'hallucination':
            key_metric_value = result.factual_accuracy
            success = key_metric_value >= thresholds['hallucination_accuracy']
        elif test_name == 'socratic':
            key_metric_value = result.socratic_accuracy
            success = key_metric_value >= thresholds['socratic_accuracy']
        elif test_name == 'gpu_stress':
            key_metric_value = result.get('max_stable_concurrency', 0)
            success = key_metric_value >= 5  # Require at least 5 concurrent users
        elif test_name == 'reference_accuracy':
            key_metric_value = result.reference_accuracy
            success = key_metric_value >= 80.0  # 80% accuracy threshold
        elif test_name == 'ontrack_tasks':
            key_metric_value = result.retrieval_accuracy
            success = key_metric_value >= 80.0  # 80% accuracy threshold

        # Log the primary result of the test
        logging.info(f"{test_name.replace('_', ' ').title()} result: {key_metric_value:.2f}")

        return {
            'test_type': test_name,
            'success': success,
            'details': result if isinstance(result, dict) else result.__dict__
        }

    def run_single_test(self, test_name: str) -> Dict[str, Any]:
        """Runs a single, specified test."""
        self.start_time = time.time()
        
        test_map = {
            'off-topic': test_off_topic.run_test,
            'cold-start': test_cold_start.run_test,
            'memory': test_memory.run_test,
            'isolation': test_isolation.run_test,
            'performance': test_performance.run_test,
            'hallucination': test_hallucination.run_test,
            'socratic': test_socratic.run_test,
            'gpu_stress': test_gpu_stress.run_test,
            'reference_accuracy': test_reference_accuracy.run_test,
            'ontrack_tasks': test_ontrack_tasks.run_test
        }
        
        if test_name not in test_map:
            raise ValueError(f"Unknown test '{test_name}'.")
        
        if test_name != 'quick':
            if not self.check_system_health().get('healthy'):
                raise Exception("System health check failed. Aborting test.")
            self.start_gpu_monitoring()
        
        try:
            logging.info(f"▶️  Running single test: {test_name}")
            result = self._run_and_process_test(test_name, test_map[test_name])
            
            execution_time = time.time() - self.start_time
            report = {
                "test_name": test_name,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "success": result.get('success', False),
                "gpu_log": self.gpu_log_path if test_name != 'quick' else None
            }
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"results/{test_name}_test_{ts}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.info(f"📊 Test result saved to: {report_path}")
            logging.info(f"✅ Test '{test_name}' completed in {execution_time:.1f}s")
            return report
            
        finally:
            if test_name != 'quick':
                self.stop_gpu_monitoring()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Runs the complete test suite."""
        self.start_time = time.time()
        logging.info("🚀 Starting AI Tutor Test Suite (Full)")
        
        self.start_gpu_monitoring()
        
        try:
            if not self.check_system_health().get('healthy'):
                raise Exception("System health check failed. Aborting test suite.")
            
            test_map = {
                'off_topic': test_off_topic.run_test,
                'cold_start': test_cold_start.run_test,
                'memory': test_memory.run_test,
                'isolation': test_isolation.run_test,
                'performance': test_performance.run_test,
                'hallucination': test_hallucination.run_test,
                'socratic': test_socratic.run_test,
                'gpu_stress': test_gpu_stress.run_test,
                'reference_accuracy': test_reference_accuracy.run_test,
                'ontrack_tasks': test_ontrack_tasks.run_test
            }

            for name, func in test_map.items():
                self.results[name] = self._run_and_process_test(name, func)
            
            return self.generate_final_report()
            
        except Exception as e:
            logging.error(f"Test suite failed with an exception: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
            
        finally:
            self.stop_gpu_monitoring()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generates a comprehensive final report for the full test suite."""
        execution_time = time.time() - (self.start_time or time.time())
        
        tests_passed = sum(1 for r in self.results.values() if r.get('success', False))
        total_tests = len(self.results)
        overall_success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        supervisor_summary = {
            "overall_status": "READY FOR REVIEW" if overall_success_rate >= 60 else "NEEDS IMPROVEMENT",
            "tests_passed": f"{tests_passed}/{total_tests}",
            "success_rate": f"{overall_success_rate:.1f}%",
            "key_findings": {
                "off_topic_detection_accuracy": self.results.get('off_topic', {}).get('details', {}).get('detection_accuracy', 0),
                "memory_accuracy": self.results.get('memory', {}).get('details', {}).get('memory_accuracy', 0),
                "avg_response_time_seconds": self.results.get('performance', {}).get('details', {}).get('average_response_time', 0),
                "session_isolation_score": self.results.get('isolation', {}).get('details', {}).get('isolation_score', 0),
                "factual_accuracy": self.results.get('hallucination', {}).get('details', {}).get('factual_accuracy', 0),
                "socratic_accuracy": self.results.get('socratic', {}).get('details', {}).get('socratic_accuracy', 0),
                "gpu_stress_score": self.results.get('gpu_stress', {}).get('details', {}).get('peak_gpu_util', 0),
                "ontrack_tasks_accuracy": self.results.get('ontrack_tasks', {}).get('details', {}).get('retrieval_accuracy', 0)
            },
            "recommendations": self._generate_recommendations()
        }
        
        report = {
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time_seconds": execution_time,
                "base_url": self.base_url,
                "gpu_log_file": self.gpu_log_path
            },
            "supervisor_summary": supervisor_summary,
            "detailed_results": self.results,
            "thresholds_used": self.context.thresholds
        }
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"results/comprehensive_report_{ts}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"📊 Comprehensive report saved to: {report_path}")
        logging.info(f"🎉 Test suite completed in {execution_time:.1f}s")
        logging.info(f"📈 Overall success rate: {overall_success_rate:.1f}%")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generates targeted recommendations based on test results."""
        recommendations = []
        thresholds = self.context.thresholds
        
        if not self.results.get('off_topic', {}).get('success'):
            recommendations.append("🎯 Off-Topic Detection: Accuracy is below threshold. Review RAG retrieval strategy.")
        if not self.results.get('memory', {}).get('success'):
            recommendations.append("🧠 Conversation Memory: Accuracy is low. Consider expanding the context window.")
        if not self.results.get('performance', {}).get('success'):
            recommendations.append("⚡ Performance: Response time or success rate is below threshold. Investigate model optimization or hardware.")
        if not self.results.get('isolation', {}).get('success'):
            recommendations.append("🔒 Session Isolation: Score is below the critical threshold. This is a high-priority issue.")
        if not self.results.get('cold_start', {}).get('success'):
            recommendations.append("🌡️ Cold Start: Performance degradation is significant. Implement a model warming strategy.")
        if not self.results.get('hallucination', {}).get('success'):
            recommendations.append("🔍 Factual Accuracy: Hallucination rate is high. Refine the RAG retrieval process.")
        if not self.results.get('socratic', {}).get('success'):
            recommendations.append("🧠 Socratic Responses: The tutor is not asking enough guiding questions. Review the prompt to encourage more Socratic interaction.")
        if not self.results.get('gpu_stress', {}).get('success'):
            recommendations.append("🎮 GPU Stress: GPU utilization is high. Consider reducing the number of concurrent sessions or optimizing model parameters.")
        if not self.results.get('ontrack_tasks', {}).get('success'):
            recommendations.append("📋 OnTrack Tasks: Accuracy is below threshold. Review the OnTrack task retrieval process.")

        if not recommendations:
            recommendations.append("✅ Excellent! All tests passed within their acceptable thresholds.")
        
        return recommendations


def main():
    """Main entry point for the test runner script."""
    parser = argparse.ArgumentParser(
        description="AI Tutor Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: You must `conda activate test_app` before running any test.
Examples:
  python test_runner.py                         # Run all tests
  python test_runner.py --test off-topic        # Test off-topic detection only
  python test_runner.py --test memory           # Test memory functionality only
  python test_runner.py --test performance      # Test performance only
  python test_runner.py --test isolation        # Test session isolation only
  python test_runner.py --test cold-start       # Test cold start vs warm start
  python test_runner.py --test hallucination    # Test hallucination detection only
  python test_runner.py --test socratic         # Test Socratic response only
  python test_runner.py --test gpu_stress       # Test GPU stress only
  python test_runner.py --test reference_accuracy # Test Reference Accuracy
  python test_runner.py --test ontrack_tasks    # Test OnTrack task retrieval only
        """
    )
    
    parser.add_argument(
        '--test', 
        choices=['off-topic', 'cold-start', 'memory', 'isolation', 'performance', 'hallucination', 'hallucination_single', 'socratic', 'gpu_stress', 'reference_accuracy', 'ontrack_tasks'],
        help="Run a specific test instead of the full suite."
    )
    
    parser.add_argument(
        '--url',
        default="http://localhost:8000",
        help="Base URL for the AI Tutor API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    runner = TestRunner(args.url)
    
    try:
        if args.test:
            report = runner.run_single_test(args.test)
            
            print(f"\n{'='*50}")
            print(f"SINGLE TEST RESULT: {args.test.upper()}")
            print(f"{'='*50}")
            print(f"Success: {'✅' if report.get('success') else '❌'}")
            print(f"Execution Time: {report.get('execution_time', 0):.2f}s")
            print(f"{'='*50}")
            
        else:
            final_report = runner.run_all_tests()
            
            print("\n" + "="*60)
            print("FINAL SUMMARY FOR SUPERVISORS")
            print("="*60)
            summary = final_report.get('supervisor_summary', {})
            print(f"  Status: {summary.get('overall_status')}")
            print(f"  Tests Passed: {summary.get('tests_passed')}")
            print(f"  Success Rate: {summary.get('success_rate')}")
            print("\n  Key Metrics:")
            for metric, value in summary.get('key_findings', {}).items():
                print(f"    - {metric.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"    - {metric.replace('_', ' ').title()}: {value}")
            print("\n  Recommendations:")
            for rec in summary.get('recommendations', []):
                print(f"    • {rec}")
            print("="*60)
            
    except KeyboardInterrupt:
        logging.info("\nTest suite interrupted by user.")
    except Exception as e:
        logging.critical(f"\nAn unrecoverable error occurred: {e}")


if __name__ == "__main__":
    main()