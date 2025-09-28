## 🧠 AI-Tutor
AI-Tutor is a RAG (Retrieval-Augmented Generation) pipeline designed to turn unstructured document collections into intelligent, queryable knowledge systems. It leverages modern LLM tooling for smart document ingestion, indexing, and frontend integration.

## 📦 Prerequisites
Before running the project, make sure the following dependencies are installed:

### System Requirements
- Python 3.10+ (managed by conda)
- Node.js 18+ (for frontend)
- Anaconda3/conda (for Python environment)

### Quick Setup
```bash
# Backend dependencies (one command)
cd AI-Tutor
conda env create -f environment.yml
conda activate test_app

# Frontend dependencies (one command)  
cd AI-tutor-frontend
npm install
```

**Note:** The `environment.yml` and `package.json` files contain all exact dependency versions. No manual installation needed!

## ⚙️ How It Works
* **Document Conversion**
    - Input documents (in various formats) are parsed and converted into markdown files using Docling.

* **Vector DB Construction**
    - These markdown files are embedded and stored in a ChromaDB vector database.

* **RAG Application**
    - A LangChain-based pipeline is built to query the vector store using LLMs for intelligent response generation.

* **API Integration**
    - A FastAPI backend wraps the RAG logic for integration with the frontend application.

**Architecture Overview**
* **Overall architecture** 
![alt text](AI-Tutor/rag_pipline.png)

## ⚙️ First-Time Setup

### For New Developers

1. **Copy configuration template:**
   ```bash
   cp config.env.template config.env
   ```

2. **Edit config.env with your details:**
   
   Update these values:
   - `BACKEND_DIR` - Your AI-Tutor backend directory path
   - `FRONTEND_DIR` - Your AI-tutor-frontend/client directory path  
   - `USERNAME` - Your Deakin username
   - `FRONTEND_SERVER` - Usually "ai-tutor.ai.deakin.edu.au"
   - `FRONTEND_HOST` - Usually "10.72.191.84"

3. **Test the configuration is valid:**
   ```bash
   ./ai-tutor.sh
   ```

**Note:** Contact your supervisor (or the University IT department) for your correct directory paths and server details.

## 🚀 Quick Start

### For New Developers
```bash
# Start everything (persistent tmux sessions)
./ai-tutor.sh full

# Access the application
# 🌐 Frontend (public): http://10.72.191.84:5000
# 🔧 Backend (local):   http://localhost:8000  
# 📚 API Docs:          http://127.0.0.1:8000/docs

# Check status
./ai-tutor.sh status

# Stop everything when done
./ai-tutor.sh stop
```

### For Development/Testing
```bash
# Backend only (persistent tmux)
./ai-tutor.sh backend

# Quick restart (no full model reload)
./ai-tutor.sh restart

# Run comprehensive tests
python run_tests.py
```

### Available Commands
Run `./ai-tutor.sh` to see all available commands and their descriptions.

**⚠️ Important:** 
- Backend startup may take minutes to load the 27GB microsoft/phi-4 model
- Services run in **persistent tmux sessions** - survive SSH disconnection
- Use `tmux attach -t ai-backend` to monitor backend startup progress
- Use `tmux attach -t ai-frontend` to monitor frontend

## 🔧 API Endpoints

Visit `http://127.0.0.1:8000/docs` for complete interactive documentation.

### Interactive API Documentation
FastAPI automatically serves interactive documentation:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`  
- **OpenAPI JSON**: `http://127.0.0.1:8000/openapi.json`

Main endpoints:
- **`GET /health`** - Health check and system status  
- **`POST /query`** - Main chat endpoint (requires `query`, `session_id`)
- **`GET /quiz/weeks`** - Get available quiz weeks
- **`POST /quiz/generate`** - Generate quiz questions for a week

## ⚙️ Environment Setup

```bash
# Navigate to backend directory
cd AI-Tutor

# Create conda environment
module load Anaconda3
conda env create -f environment.yml
conda activate test_app
```

## 🧪 Testing

The AI-Tutor includes a comprehensive test suite that can be run from anywhere in the project using the `run_tests.py` script.

### Quick Start
```bash
# Activate the test environment first
conda activate test_app

# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --test memory           # Test memory functionality and persistence
python run_tests.py --test off-topic        # Test off-topic detection capabilities
python run_tests.py --test performance      # Test system performance and response times
python run_tests.py --test isolation        # Test session isolation and data separation
python run_tests.py --test cold-start       # Test cold start vs warm start performance
python run_tests.py --test hallucination    # Test hallucination detection mechanisms
python run_tests.py --test socratic         # Test Socratic questioning methodology
python run_tests.py --test gpu_stress       # Test GPU resource utilisation
python run_tests.py --test reference_accuracy # Test reference accuracy and citation quality
python run_tests.py --test ontrack_tasks    # Test OnTrack task retrieval functionality
```

**Note:** The `run_tests.py` script allows you to run tests from anywhere in the project directory. It automatically navigates to the test directory and executes the appropriate test runner.

**Note** The comprehensive test dataset (`_my_tests/golden_dataset.json`) contains all JSON test cases used by the test scripts when querying the API, including multi-turn conversations, session isolation scenarios, and performance benchmarks for the SIT796 Reinforcement Learning course. These tests were based, in part, off usage history of students.

### Test Results
Test results are automatically saved to `_my_tests/results/` with timestamped filenames:
- **JSON files**: Detailed test outputs for each test category (memory, performance, GPU stress, etc.)
- **CSV files**: GPU monitoring data including utilization, memory usage, and power consumption
- **File naming**: `{test_type}_test_{YYYYMMDD}_{HHMMSS}.json` or `gpu_{YYYYMMDD}_{HHMMSS}.csv`

Results are excluded from git to avoid repository bloat, but can be analyzed locally for performance tracking and debugging.

### Test Infrastructure
The test suite is built on a comprehensive measurement system (`_my_tests/measurement_system.py`) that provides:
- Semantic similarity evaluation using the same Nomic model as the RAG system
- Standardized test contexts and result data structures
- Performance thresholds and benchmarking capabilities
- Golden dataset integration for consistent test scenarios

## 📚 Development Documentation

The `_my_notes/Dev/` folder contains comprehensive development notes including:
- Performance optimization strategies
- Behavior analysis and improvements
- Technical debt and future enhancements
- Testing methodologies
- Architecture decisions

**Key files for new developers:**
- `_my_notes/Dev/_Todo.md` - Current work items and priorities
- `_my_notes/Dev/Behavior.md` - System behavior analysis
- `_my_notes/Dev/1_Memory.md` - Memory management strategies
- `_my_notes/Dev/2_Performance.md` - Performance optimization notes


## Database
The pre-built RAG database (`AI-Tutor/RL_db_reference_1k_500/`) is included in this repository. New developers can start using the AI-Tutor immediately without rebuilding the database.

To rebuild the database from scratch (if needed):
```bash
cd AI-Tutor
python docling_run.py --input-directory ./grouped_weeks --vector_db ./RL_db_reference_1k_500
```
