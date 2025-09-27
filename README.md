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
- Backend startup may take minute to load the 27GB microsoft/phi-4 model
- Services run in **persistent tmux sessions** - survive SSH disconnection
- Use `tmux attach -t ai-backend` to monitor backend startup progress
- Use `tmux attach -t ai-frontend` to monitor frontend

## 🔧 API Endpoints

Visit `http://127.0.0.1:8000/docs` for complete interactive documentation.

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

```bash
# Run full test suite
python run_tests.py

# Run specific test categories
cd _my_tests
python test_runner.py --category performance
python test_runner.py --category memory
```

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
