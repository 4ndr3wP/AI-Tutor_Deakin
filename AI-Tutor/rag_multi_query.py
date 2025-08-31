"""
Multi-turn RAG backend for Deakin AI-Tutor
------------------------------------------

• Same model/embedding/vector-DB settings as rag_single_query.py
• Adds ConversationBufferWindowMemory for chat history
• One chain per session to avoid user-data bleeding
"""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from langchain_community.llms import VLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
class CFG:
    # Model settings
    MODEL_NAME = "microsoft/phi-4"
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    PERSIST_DIR = "./RL_db_reference_1k_500"
    
    # Generation parameters
    MAX_NEW_TOKENS = 1024 # 3-4 paragraphs of text
    TEMPERATURE = 0.2
    TOP_P = 0.9
    GPU_FRACTION = 0.5
    
    # Retrieval & memory
    DEFAULT_K = 5
    MEMORY_TURNS = 10
    
    # Performance
    THREAD_WORKERS = 4
    
    # Timeout settings
    REQUEST_TIMEOUT = 300.0  # 5 minutes


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ai-tutor-multi")


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbeddings(Embeddings):
    """Optimized embedding wrapper for SentenceTransformer."""
    
    def __init__(self, model_name: str = CFG.EMBEDDING_MODEL, batch_size: int = 64):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        log.info(f"Loading embeddings on {self.device}")
        self.model = SentenceTransformer(
            model_name, 
            device=self.device, 
            trust_remote_code=True
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {text}" for text in texts]
        return self.model.encode(
            prefixed, 
            convert_to_tensor=False, 
            batch_size=self.batch_size
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            f"search_query: {text}", 
            convert_to_tensor=False
        ).tolist()


# --------------------------------------------------------------------------- #
# Prompt Template                                                             #
# --------------------------------------------------------------------------- #
RAG_TEMPLATE = """<|im_start|>system<|im_sep|>
You are an AI tutor assisting with university unit content. You will be given
course context, chat history, and a student question.

1. Use ONLY the context and history to answer factually and clearly.
2. Stay concise and on-topic.
3. If question is outside the context, give a brief overview answer and do not hallucinate.
4. After the answer, provide a list of relevant weeks or slides in square brackets,
   e.g., Related material: [week 1, week 2]. Do NOT reveal full document text.
5. If malicious or sensitive, REFUSE.

<|im_start|>user<|im_sep|>
Conversation history:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
<|im_start|>assistant<|im_sep|>
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=RAG_TEMPLATE,
)


# --------------------------------------------------------------------------- #
# Multi-turn Manager                                                          #
# --------------------------------------------------------------------------- #
class MultiTurnManager:
    """Manages LLM, vectorstore, and per-session memory."""

    def __init__(self):
        log.info("Initializing multi-turn RAG system...")
        
        # Initialize LLM
        self.llm = VLLM(
            model=CFG.MODEL_NAME,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_new_tokens=CFG.MAX_NEW_TOKENS,
            temperature=CFG.TEMPERATURE,
            top_p=CFG.TOP_P,
            gpu_memory_utilization=CFG.GPU_FRACTION,
        )
        log.info("✅ LLM ready")

        # Initialize embeddings & vectorstore
        self.embeddings = SentenceTransformerEmbeddings()
        self.vdb = Chroma(
            persist_directory=CFG.PERSIST_DIR,
            embedding_function=self.embeddings,
        )
        log.info("✅ Vectorstore ready")

        # LLM chain
        self.chain = LLMChain(llm=self.llm, prompt=PROMPT)
        
        # Session management
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Thread pool for async retrieval
        self.executor = ThreadPoolExecutor(max_workers=CFG.THREAD_WORKERS)
        
        log.info("✅ Multi-turn system ready")

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Thread-safe session memory factory."""
        with self._locks[session_id]:
            if session_id not in self._memories:
                self._memories[session_id] = ConversationBufferWindowMemory(
                    k=CFG.MEMORY_TURNS,
                    memory_key="chat_history",
                    return_messages=True,
                )
                log.info(f"Created memory for session {session_id[:8]}...")
            return self._memories[session_id]

    async def _retrieve_docs(self, question: str, k: int = CFG.DEFAULT_K) -> str:
        """Async document retrieval."""
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            self.executor,
            lambda: self.vdb.similarity_search(question, k=k)
        )
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _format_history(messages: List) -> str:
        """Format chat history for prompt."""
        if not messages:
            return "No previous conversation."
        
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage) or getattr(msg, 'type', '') == 'human':
                lines.append(f"Student: {msg.content}")
            elif isinstance(msg, AIMessage) or getattr(msg, 'type', '') == 'ai':
                lines.append(f"AI Tutor: {msg.content}")
        
        return "\n".join(lines)

    async def ask(self, question: str, session_id: str, k: Optional[int] = None) -> str:
        """Generate response with memory."""
        try:
            # Get session memory
            memory = self._get_memory(session_id)
            
            # Retrieve documents (async)
            context = await self._retrieve_docs(question, k or CFG.DEFAULT_K)
            
            # Format chat history
            memory_vars = memory.load_memory_variables({})
            history = self._format_history(memory_vars.get("chat_history", []))
            
            log.info(f"Session {session_id[:8]}: {len(memory_vars.get('chat_history', []))} previous messages")
            
            # Generate response
            response = await self.chain.arun(
                context=context,
                question=question,
                chat_history=history
            )
            
            # Save to memory
            memory.save_context({"input": question}, {"output": response})
            
            return response
            
        except Exception as e:
            log.error(f"Failed to generate response: {e}")
            log.error(traceback.format_exc())
            return "I'm experiencing technical difficulties. Please try again."

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# FastAPI Application                                                         #
# --------------------------------------------------------------------------- #
manager: Optional[MultiTurnManager] = None

app = FastAPI(title="AI-Tutor Multi-turn", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global manager
    manager = MultiTurnManager()
    yield
    if manager and hasattr(manager, 'executor'):
        manager.executor.shutdown(wait=True)


app.router.lifespan_context = lifespan


# --------------------------------------------------------------------------- #
# API Models & Routes                                                         #
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str
    session_id: str
    k: Optional[int] = None


class QueryResponse(BaseModel):
    response: str
    session_id: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if manager else "initializing",
        "model": CFG.MODEL_NAME,
        "memory_turns": CFG.MEMORY_TURNS
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main chat endpoint."""
    if not manager:
        return QueryResponse(
            response="System initializing – please retry in a moment.",
            session_id=request.session_id,
        )

    query = request.query.strip()
    if not query:
        return QueryResponse(
            response="What would you like to learn about?",
            session_id=request.session_id,
        )

    try:
        response = await manager.ask(query, request.session_id, request.k)
        return QueryResponse(response=response, session_id=request.session_id)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Endpoint error: {e}")
        return QueryResponse(
            response="Unexpected server error. Please try again.",
            session_id=request.session_id,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
