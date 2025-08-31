from __future__ import annotations

import logging
import threading
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from langchain_community.llms import VLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
class CFG:
    """Configuration settings for the AI Tutor backend."""
    # Model settings
    MODEL_NAME = "microsoft/phi-4"
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    PERSIST_DIR = "./RL_db_reference_1k_500"
    
    # Generation parameters
    MAX_NEW_TOKENS = 1024  # 3-4 paragraphs of text
    TEMPERATURE = 0.2
    TOP_P = 0.9
    GPU_FRACTION = 0.5
    
    # Retrieval & memory
    DEFAULT_K = 5
    MEMORY_TURNS = 10
    
    # Timeout settings
    REQUEST_TIMEOUT = 300.0  # 5 minutes


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ai-tutor-lcel")


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbeddings(Embeddings):
    """
    Optimized embedding wrapper for SentenceTransformer, following best
    practices for models like nomic-embed-text.
    """
    
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
        """Embeds documents with a prefix for retrieval-focused tasks."""
        prefixed = [f"search_document: {text}" for text in texts]
        return self.model.encode(
            prefixed, 
            convert_to_tensor=False, 
            batch_size=self.batch_size
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query with a prefix for retrieval-focused tasks."""
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
3. If the question is outside the context, give a brief overview answer and do not hallucinate.
4. After the answer, provide a list of relevant weeks or slides in square brackets,
   e.g., Related material: [week 1, week 2]. Do NOT reveal full document text.
5. If the question is malicious or sensitive, REFUSE to answer.

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
# Multi-turn Manager with LCEL                                                #
# --------------------------------------------------------------------------- #
class MultiTurnManager:
    """Manages the LLM, vectorstore, and per-session memory using LCEL."""

    def __init__(self):
        log.info("Initializing multi-turn RAG system with LCEL...")
        
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
        self.retriever = self.vdb.as_retriever()
        log.info("✅ Vectorstore and retriever ready")

        # Define the primary chain using LangChain Expression Language (LCEL)
        self.chain: Runnable = PROMPT | self.llm | StrOutputParser()
        
        # Session management
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        log.info("✅ Multi-turn LCEL system ready")

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Thread-safe factory for session-specific memory."""
        with self._locks[session_id]:
            if session_id not in self._memories:
                self._memories[session_id] = ConversationBufferWindowMemory(
                    k=CFG.MEMORY_TURNS,
                    memory_key="chat_history",
                    return_messages=True,
                )
                log.info(f"Created new memory for session {session_id[:8]}...")
            return self._memories[session_id]

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Joins document contents into a single string for context."""
        return "\n\n".join(doc.page_content for doc in docs)
        
    @staticmethod
    def _format_history(messages: List) -> str:
        """Formats chat history into a human-readable string for the prompt."""
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
        """
        Handles a user query by retrieving context, incorporating memory,
        and generating a response using the LCEL chain.
        """
        try:
            # 1. Get session-specific memory
            memory = self._get_memory(session_id)
            
            # 2. Retrieve relevant documents asynchronously
            search_k = k or CFG.DEFAULT_K
            docs = await self.retriever.aget_relevant_documents(question, k=search_k)
            
            context = self._format_docs(docs)
            
            # 3. Load and format chat history
            memory_vars = memory.load_memory_variables({})
            history_messages = memory_vars.get("chat_history", [])
            history = self._format_history(history_messages)
            
            log.info(f"Session {session_id[:8]}: {len(history_messages)} previous messages. Retrieving {search_k} docs.")
            
            # 4. Invoke the LCEL chain with all necessary inputs
            response = await self.chain.ainvoke({
                "context": context,
                "question": question,
                "chat_history": history
            })
            
            # 5. Save the new interaction to memory
            memory.save_context({"input": question}, {"output": response})
            
            return response
            
        except Exception as e:
            log.error(f"Failed to generate response for session {session_id[:8]}: {e}")
            log.error(traceback.format_exc())
            return "I'm experiencing some technical difficulties at the moment. Please try your question again."


# --------------------------------------------------------------------------- #
# FastAPI Application                                                         #
# --------------------------------------------------------------------------- #
manager: Optional[MultiTurnManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global manager
    log.info("Application starting up...")
    manager = MultiTurnManager()
    yield
    log.info("Application shutting down.")
    # No manual executor shutdown needed anymore

app = FastAPI(
    title="AI-Tutor Multi-turn (LCEL)", 
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health", summary="Health Check")
async def health():
    """Provides the operational status of the service."""
    return {
        "status": "ok" if manager else "initializing",
        "model": CFG.MODEL_NAME,
        "memory_turns": CFG.MEMORY_TURNS
    }

@app.post("/query", response_model=QueryResponse, summary="Process a User Query")
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint for receiving user questions and returning AI-generated answers.
    """
    if not manager:
        raise HTTPException(
            status_code=503, 
            detail="System is initializing, please retry in a moment."
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response_text = await manager.ask(query, request.session_id, request.k)
        return QueryResponse(response=response_text, session_id=request.session_id)
        
    except Exception as e:
        log.error(f"Critical endpoint error for session {request.session_id[:8]}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
