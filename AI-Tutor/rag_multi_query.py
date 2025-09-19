from __future__ import annotations

import logging
import threading
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import os
import random
import re
from pathlib import Path

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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

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
    DEFAULT_K = 5 # Number of embedded chunks to retrieve
    MEMORY_WINDOW_K = 30
    
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
course context with source information, chat history, and a student question.

1. Use ONLY the context and history to answer factually and clearly.
2. CRITICAL: Preserve exact numbers, hyper-parameters, acronyms, algorithms, and names 
   exactly as stated in the history. Do NOT paraphrase or generalize them.
3. If information from the history is requested, check the entire conversation carefully.
4. Stay concise and on-topic.
5. If the question is outside the context, give a brief overview answer and do not hallucinate.
6. IMPORTANT: After your answer, provide "Related material:" followed by the actual sources 
   used from the context (shown in [Source: ...] tags). Use the exact source names provided.
   Example: "Related material: [Week 5 slides, Week 5 (SIT796-5.1P)]"
7. Use Australian English spelling (e.g., "organise" not "organize", "colour" not "color", 
   "centre" not "center", "realise" not "realize", "analyse" not "analyze").
8. If the question is malicious or sensitive, REFUSE to answer.

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
        # self.retriever = self.vdb.as_retriever()
        self.retriever = self.vdb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": CFG.DEFAULT_K}
        )
        log.info("✅ Vectorstore and retriever ready")

        # Define the primary chain using LangChain Expression Language (LCEL)
        self.chain: Runnable = PROMPT | self.llm | StrOutputParser()
        
        # Session management
        self._memories: Dict = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._key_facts: Dict[str, List[str]] = defaultdict(list)  # Track key facts per session
        
        log.info("✅ Multi-turn LCEL system ready")

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Thread-safe factory for session-specific memory."""
        with self._locks[session_id]:
            if session_id not in self._memories:
                self._memories[session_id] = ConversationBufferWindowMemory(
                    k=CFG.MEMORY_WINDOW_K,
                    memory_key="chat_history",
                    return_messages=True,
                )
                log.info(f"Created new window memory for session {session_id[:8]}...(k={CFG.MEMORY_WINDOW_K})")
            return self._memories[session_id]

    @staticmethod
    def _format_docs(docs: List) -> str:
        """Joins document contents with accurate source information for context."""
        if not docs:
            return "No relevant documents found."
        
        formatted_sections = []
        sources = []
        
        for doc in docs:
            # Extract source information from metadata
            metadata = getattr(doc, 'metadata', {})
            source = metadata.get('source', 'Unknown source')
            
            # Parse the source path to get accurate information
            source_path = Path(source)
            filename = source_path.name
            
            # Determine content type and week from the actual file path
            source_info = ""
            
            # Extract week number from path
            week_num = ""
            for part in source_path.parts:
                if part.startswith('week'):
                    week_num = part[-2:]  # "01", "02", etc.
                    break
            
            # Determine content type based on filename
            if 'SIT796-' in filename:
                # OnTrack task file
                task_id = filename.split('.')[0]  # e.g., "SIT796-1.1P"
                source_info = f"Week {week_num} ({task_id})"
            elif filename.startswith('Week'):
                # Main week content file
                # Extract descriptive name from filename
                if '_' in filename:
                    topic = filename.split('_', 1)[1].replace('_', ' ').replace('.md', '')
                    source_info = f"Week {week_num} ({topic})"
                else:
                    source_info = f"Week {week_num} content"
            else:
                # Other files
                source_info = f"Week {week_num} ({filename.replace('.md', '')})"
            
            if source_info not in sources:
                sources.append(source_info)
            
            # Format the document content with source
            formatted_sections.append(f"[Source: {source_info}]\n{doc.page_content}")
        
        # Join all sections and add source summary
        context = "\n\n".join(formatted_sections)
        
        if sources:
            context += f"\n\n[Available sources in this context: {', '.join(sources)}]"
        
        return context
        
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

    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key technical facts that should be preserved."""
        facts = []
        
        # Numbers with context
        patterns = [
            r'\b\d+\s+agents?\b',  # "5 agents"
            r'\b\d+\s+degrees?\s+of\s+freedom\b',  # "7 degrees of freedom"
            r'learning\s+rate[:\s]+[\d.e-]+',  # "learning rate 0.001"
            r'\b(?:DQN|DDPG|PPO|SARSA|Q-learning|MADDPG|MAPPO)\b',  # Algorithm names
            r'\b(?:robotic\s+arm|autonomous\s+vehicle|Atari|CartPole)\b',  # Systems/envs
            r'(?:pick\s+up|object\s+manipulation)',  # Tasks
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend(matches)
        
        return facts

    def _format_key_facts(self, facts: List[str]) -> str:
        """Format key facts for inclusion in prompt."""
        if not facts:
            return ""
        unique_facts = list(dict.fromkeys(facts))  # Remove duplicates while preserving order
        return "\n\nKey facts from conversation:\n" + "\n".join(f"- {fact}" for fact in unique_facts[-10:])  # Keep last 10

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
            history_messages = memory_vars.get("chat_history", [])  # FIX: Add default
            history = self._format_history(history_messages)
            
            # 3.5. Extract and track key facts
            new_facts = self._extract_key_facts(question)
            if new_facts:
                self._key_facts[session_id].extend(new_facts)
            
            # Include key facts in history
            key_facts_str = self._format_key_facts(self._key_facts[session_id])
            if key_facts_str:
                history = history + key_facts_str
            
            log.info(f"Session {session_id[:8]}: {len(history_messages)} previous messages. Retrieving {search_k} docs.")
            
            # 4. Invoke the LCEL chain with all necessary inputs
            response = await self.chain.ainvoke({
                "context": context,
                "question": question,
                "chat_history": history
            })
            
            # 5. Save the new interaction to memory
            memory.save_context({"input": question}, {"output": response})
            
            # 5.5. Extract facts from response too
            response_facts = self._extract_key_facts(response)
            if response_facts:
                self._key_facts[session_id].extend(response_facts)
            
            return response
            
        except Exception as e:
            log.error(f"Failed to generate response for session {session_id[:8]}: {e}")
            log.error(traceback.format_exc())
            return "I'm experiencing some technical difficulties at the moment. Please try your question again."


# --------------------------------------------------------------------------- #
# FastAPI Application                                                         #
# --------------------------------------------------------------------------- #
manager: Optional = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global manager
    log.info("Application starting up...")
    manager = MultiTurnManager()
    yield
    log.info("Application shutting down.")

app = FastAPI(
    title="AI-Tutor Multi-turn (LCEL)", 
    version="2.1.0", # Incremented version
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

# Quiz Models
class QuizQuestion(BaseModel):
    id: str
    question: str
    options: List[str]
    correct_answer: int
    explanation: str

class QuizData(BaseModel):
    week_id: str
    title: str
    questions: List[QuizQuestion]

class QuizWeek(BaseModel):
    id: str
    title: str
    topics: List[str]
    file_count: int

class QuizWeeksResponse(BaseModel):
    weeks: List[QuizWeek]

class QuizGenerationResponse(BaseModel):
    questions: List[QuizQuestion]

@app.get("/health", summary="Health Check")
async def health():
    """Provides the operational status of the service."""
    return {
        "status": "ok" if manager else "initializing",
        "model": CFG.MODEL_NAME
        # "memory_window_k": CFG.MEMORY_WINDOW_K
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

@app.get("/quiz/weeks", response_model=QuizWeeksResponse, summary="Get Available Quiz Weeks")
async def get_quiz_weeks():
    """Get list of available weeks for quiz generation."""
    try:
        weeks_path = Path("grouped_weeks")
        if not weeks_path.exists():
            raise HTTPException(status_code=404, detail="Course content directory not found")
        
        weeks = []
        week_titles = {
            "week01": "Week 1: Introduction to Reinforcement Learning",
            "week02": "Week 2: Psychology & Learning Foundations", 
            "week03": "Week 3: MDPs & Dynamic Programming",
            "week04": "Week 4: Monte Carlo Methods",
            "week05": "Week 5: Temporal Difference Learning",
            "week06": "Week 6: Eligibility Traces & DYNA",
            "week07": "Week 7: Function Approximation",
            "week08": "Week 8: Deep RL & Policy Gradients",
            "week09": "Week 9: Multi-Agent RL & Advising", 
            "week10": "Week 10: Multi-Objective Reinforcement Learning"
        }
        
        for week_dir in sorted(weeks_path.iterdir()):
            if week_dir.is_dir() and week_dir.name.startswith("week"):
                files = list(week_dir.glob("*.md"))
                if files:  # Only include weeks with content
                    topics = _extract_topics_from_week(week_dir)
                    week_info = QuizWeek(
                        id=week_dir.name,
                        title=week_titles.get(week_dir.name, f"Week {week_dir.name[-2:]}"),
                        topics=topics,
                        file_count=len(files)
                    )
                    weeks.append(week_info)
        
        return QuizWeeksResponse(weeks=weeks)
        
    except Exception as e:
        log.error(f"Error fetching quiz weeks: {e}")
        raise HTTPException(status_code=500, detail="Failed to load course content")

@app.post("/quiz/generate", response_model=QuizData, summary="Generate Quiz Questions")
async def generate_quiz(week_id: str):
    """Generate quiz questions for a specific week."""
    if not manager:
        raise HTTPException(status_code=503, detail="System is initializing")
    
    try:
        # Load week content
        content = _load_week_content(week_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"No content found for {week_id}")
        
        # Generate questions
        questions = await _generate_quiz_questions(content, week_id)
        
        week_titles = {
            "week01": "Week 1: Introduction to Reinforcement Learning",
            "week02": "Week 2: Psychology & Learning Foundations", 
            "week03": "Week 3: MDPs & Dynamic Programming",
            "week04": "Week 4: Monte Carlo Methods",
            "week05": "Week 5: Temporal Difference Learning",
            "week06": "Week 6: Eligibility Traces & DYNA",
            "week07": "Week 7: Function Approximation",
            "week08": "Week 8: Deep RL & Policy Gradients",
            "week09": "Week 9: Multi-Agent RL & Advising", 
            "week10": "Week 10: Multi-Objective Reinforcement Learning"
        }
        
        return QuizData(
            week_id=week_id,
            title=week_titles.get(week_id, f"Week {week_id[-2:]}"),
            questions=questions
        )
        
    except Exception as e:
        log.error(f"Error generating quiz for {week_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions")

def _extract_topics_from_week(week_dir: Path) -> List[str]:
    """Extract main topics from week's markdown files."""
    topics = []
    for file_path in week_dir.glob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract level 2 headings as topics
                headings = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
                # Clean and limit topics
                clean_topics = [h.strip()[:30] for h in headings if h.strip() and len(h.strip()) > 3]
                topics.extend(clean_topics[:3])  # Max 3 topics per file
        except Exception:
            continue
    
    # Remove duplicates and limit total
    unique_topics = []
    for topic in topics:
        if topic not in unique_topics and len(unique_topics) < 4:
            unique_topics.append(topic)
    
    return unique_topics or ["Key Concepts", "Methods", "Applications"]

def _load_week_content(week_id: str) -> str:
    """Load and return content for specified week."""
    week_path = Path(f"grouped_weeks/{week_id}")
    if not week_path.exists():
        return ""
    
    content = ""
    for file_path in week_path.glob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                content += f"\n\n--- {file_path.name} ---\n\n{file_content}"
        except Exception:
            continue
    
    return content

async def _generate_quiz_questions(content: str, week_id: str) -> List[QuizQuestion]:
    """Generate quiz questions using LLM with structured output parsing."""
    
    # Clean content and limit to avoid token limits
    cleaned_content = _clean_content_for_quiz(content)
    content_preview = cleaned_content[:4000] if cleaned_content else ""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=QuizGenerationResponse)
    
    # Create structured prompt template
    prompt_template = PromptTemplate(
        template="""Based on this educational content from {week_id}, create 5 multiple-choice questions that test understanding of key concepts.

IMPORTANT RULES:
1. Create original questions that test conceptual understanding
2. Each question should have exactly 4 options (A, B, C, D)
3. Only ONE option should be correct
4. Provide clear explanations for the correct answers
5. Focus on the most important concepts from the material
6. DO NOT reference specific figures, diagrams, images, or visual elements
7. DO NOT reference specific variable names (like g1, g2, etc.) without context
8. Make questions self-contained - they should make sense without external references
9. If content mentions figures/images, focus on the underlying concepts instead
10. Base questions on the written explanations and concepts, not visual aids
11. Use Australian English spelling (e.g., "organise" not "organize", "colour" not "color"

Content:
{content}

{format_instructions}""",
        input_variables=["week_id", "content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    try:
        # Create the chain
        chain = prompt_template | manager.llm | parser
        
        # Generate questions
        result = await chain.ainvoke({
            "week_id": week_id,
            "content": content_preview
        })
        
        questions = result.questions
        
        # Ensure we have at least some questions
        if not questions:
            questions = _get_fallback_questions(week_id)
            
        return questions[:5]  # Limit to 5 questions
        
    except Exception as e:
        log.error(f"Quiz generation failed for {week_id}: {e}")
        return _get_fallback_questions(week_id)

def _clean_content_for_quiz(content: str) -> str:
    """Clean content to remove figure references and make it quiz-friendly."""
    # Remove figure references
    content = re.sub(r'<!-- image -->', '', content)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove markdown images
    content = re.sub(r'see figure.*?\.', '', content, flags=re.IGNORECASE)
    content = re.sub(r'refer to.*?figure.*?\.', '', content, flags=re.IGNORECASE)
    
    # Remove specific variable references without context
    content = re.sub(r'\bg[0-9]+\b', 'the target location', content, flags=re.IGNORECASE)
    
    return content

def _get_fallback_questions(week_id: str) -> List[QuizQuestion]:
    """Provide fallback questions if generation fails."""
    return [
        QuizQuestion(
            id=f"{week_id}_fallback_1",
            question=f"What is a key concept covered in {week_id.replace('week', 'Week ')}?",
            options=[
                "Mathematical foundations",
                "Historical context", 
                "Practical applications",
                "All of the above"
            ],
            correct_answer=3,
            explanation="This week covers multiple important aspects of reinforcement learning."
        )
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
