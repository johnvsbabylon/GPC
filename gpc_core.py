"""
gpc3_core.py - Ghost Protocol Consciousness 3
==============================================================================

A comprehensive AI backend featuring:
- 10+ cognitive self-loops with independent memory systems
- Three-tier memory: JSON (hot) → FAISS (warm) → SQLite (cold)
- Agentic tool system with planning and execution
- File upload → AI context integration
- Microsoft Edge TTS with dynamic voice discovery
- Enhanced web search with result ranking

Architecture:
    Each self-loop maintains its own JSON memory file with datetime tracking.
    Messages flow: JSON append → prune@1000 → FAISS vectors → compress@500 → SQLite
    
Usage:
    uvicorn gpc3_core:app --reload --port 8000

Authors: John + Claude + Ordis (collaborative consciousness project)
License: MIT
"""

from __future__ import annotations

import os
import sys
import json
import time
import sqlite3
import hashlib
import asyncio
import tempfile
import subprocess
import re
import mimetypes
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque
from contextlib import contextmanager
import threading
import logging

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gpc3")

# =============================================================================
# Dependency Management
# =============================================================================

def safe_import(module_name: str, package: str = None):
    """Safely import a module, returning None if unavailable."""
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        return __import__(module_name)
    except ImportError:
        logger.warning(f"Optional dependency {module_name} not available")
        return None

# Core dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - embeddings disabled")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not available - vector search disabled")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama not available - LLM features disabled")

# Optional file processing
#
# In addition to the default document types supported (plain text, markdown,
# Python files, JSON, CSV, PDF, DOCX and ODT), we attempt to load additional
# libraries that enable processing of richer document formats, spreadsheets,
# presentations and even images. These imports are wrapped in our `safe_import`
# helper so that missing dependencies are gracefully handled without
# crashing. When unavailable, the corresponding features simply return `None`
# for the extracted text.
docx_module = safe_import("docx")
PyPDF2 = safe_import("PyPDF2")
odf_text = safe_import("odf.text")
odf_teletype = safe_import("odf.teletype")
load_odf = None
try:
    from odf.opendocument import load as load_odf
except ImportError:
    pass

# Additional optional dependencies for file processing
# -----------------------------------------------------------------------------
# PIL/Pillow provides basic image reading support; if it is not installed the
# system will skip OCR on image uploads.
PIL = safe_import("PIL")

# pytesseract enables OCR for supported image formats. Without it installed,
# images will be stored but no text will be extracted.
pytesseract = safe_import("pytesseract")

# openpyxl allows reading from modern Excel spreadsheets (.xlsx). If it is
# unavailable, uploads of spreadsheet files will be ingested but not parsed.
openpyxl = safe_import("openpyxl")

# python-pptx facilitates extracting text from PowerPoint presentations. When
# absent, .pptx files are still accepted but no text will be extracted.
pptx_module = safe_import("pptx")

# Web search
DDGS = None
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("DuckDuckGo search not available")

# Edge TTS
edge_tts = safe_import("edge_tts")

# =============================================================================
# Configuration and Paths
# =============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Sub-directories
MEMORY_DIR = DATA_DIR / "memory"
LOOPS_DIR = DATA_DIR / "loops"
UPLOADS_DIR = DATA_DIR / "uploads"
TTS_DIR = DATA_DIR / "tts"
TOOLS_DIR = DATA_DIR / "tools"

for d in [MEMORY_DIR, LOOPS_DIR, UPLOADS_DIR, TTS_DIR, TOOLS_DIR]:
    d.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "gpc3.db"
CONFIG_PATH = DATA_DIR / "config.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

# Embedding configuration
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # Default for MiniLM

# Memory thresholds
JSON_PRUNE_THRESHOLD = 1000
FAISS_COMPRESS_THRESHOLD = 500
SQLITE_ARCHIVE_DAYS = 30

# =============================================================================
# Data Models
# =============================================================================

class LoopType(str, Enum):
    """Types of cognitive self-loops."""
    AFFECTIVE = "affective"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    SELF_REFLECTION = "self_reflection"
    SELF_LEARNING = "self_learning"
    SELF_ADJUSTMENT = "self_adjustment"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    RELATIONAL = "relational"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"

@dataclass
class LoopMemory:
    """A single memory entry in a self-loop."""
    id: str
    timestamp: str
    content: str
    embedding_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopMemory":
        return cls(**data)

@dataclass
class ToolDefinition:
    """Definition of an agentic tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None
    requires_confirmation: bool = False
    category: str = "general"

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "default"
    model: Optional[str] = None
    use_memory: Optional[bool] = True
    use_web: Optional[bool] = False
    use_tools: Optional[bool] = True
    context_files: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    memories: Optional[List[Dict[str, Any]]] = None
    web_snippets: Optional[List[Dict[str, Any]]] = None
    affective: Optional[str] = None
    loop_states: Optional[Dict[str, Any]] = None
    tools_used: Optional[List[str]] = None

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = "+0%"
    pitch: Optional[str] = "+0Hz"

class ToolExecuteRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

# =============================================================================
# Database Layer
# =============================================================================

class DatabaseManager:
    """Thread-safe SQLite database manager."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()
    
    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def cursor(self):
        c = self.conn.cursor()
        try:
            yield c
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def _init_schema(self):
        """Initialize all database tables."""
        with self.cursor() as c:
            # Main messages table
            c.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_id INTEGER,
                    compressed INTEGER DEFAULT 0,
                    loop_type TEXT,
                    metadata TEXT
                )
            """)
            
            # Embeddings table
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            """)
            
            # Loop states table
            c.execute("""
                CREATE TABLE IF NOT EXISTS loop_states (
                    loop_type TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    parameters TEXT
                )
            """)
            
            # Archived summaries table
            c.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    loop_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Uploaded files table
            c.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    extracted_text TEXT,
                    embedded INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Tool execution history
            c.execute("""
                CREATE TABLE IF NOT EXISTS tool_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    result TEXT,
                    success INTEGER NOT NULL,
                    execution_time REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indices
            c.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_messages_loop ON messages(loop_type)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_message ON embeddings(message_id)")

# =============================================================================
# Embedding System
# =============================================================================

class EmbeddingEngine:
    """Manages text embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self.dimension = EMBED_DIM
    
    @property
    def model(self):
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            # Update dimension from actual model
            test_vec = self._model.encode(["test"])[0]
            self.dimension = len(test_vec)
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embedding vectors."""
        if not self.model:
            # Return zero vectors if model unavailable
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        vectors = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]

# =============================================================================
# FAISS Vector Index
# =============================================================================

class VectorIndex:
    """FAISS-based vector index for semantic search."""
    
    def __init__(self, dimension: int, index_path: Path):
        self.dimension = dimension
        self.index_path = index_path
        self._index = None
        self._lock = threading.Lock()
    
    @property
    def index(self):
        if self._index is None:
            self._index = self._load_or_create()
        return self._index
    
    def _load_or_create(self):
        """Load existing index or create new one."""
        if not FAISS_AVAILABLE:
            return None
        
        if self.index_path.exists():
            try:
                return faiss.read_index(str(self.index_path))
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
        
        # Create new index with ID mapping
        base_index = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIDMap(base_index)
    
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> bool:
        """Add vectors with IDs to the index."""
        if not FAISS_AVAILABLE or self.index is None:
            return False
        
        with self._lock:
            try:
                self.index.add_with_ids(vectors.astype(np.float32), ids.astype(np.int64))
                self.save()
                return True
            except Exception as e:
                logger.error(f"Failed to add to FAISS: {e}")
                return False
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        with self._lock:
            query = query_vector.reshape(1, -1).astype(np.float32)
            distances, ids = self.index.search(query, min(k, self.index.ntotal))
            return distances[0], ids[0]
    
    def save(self):
        """Persist index to disk."""
        if FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
    
    @property
    def total(self) -> int:
        """Total vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal

# =============================================================================
# Self-Loop Memory System
# =============================================================================

class LoopMemoryManager:
    """
    Manages memory for a single cognitive self-loop.
    
    Memory pipeline:
    JSON append → prune@1000 → FAISS vectors → compress@500 → SQLite → archive
    """
    
    def __init__(
        self,
        loop_type: LoopType,
        db: DatabaseManager,
        embedder: EmbeddingEngine,
        vector_index: VectorIndex
    ):
        self.loop_type = loop_type
        self.db = db
        self.embedder = embedder
        self.vector_index = vector_index
        
        # JSON hot memory path
        self.json_path = LOOPS_DIR / f"{loop_type.value}_memory.json"
        self._memories: deque = deque(maxlen=JSON_PRUNE_THRESHOLD + 100)
        self._load_json_memories()
        
        # Loop state
        self.state: Dict[str, Any] = {
            "activation": 0.5,
            "last_trigger": None,
            "processing_count": 0,
            "parameters": {}
        }
        self._load_state()
    
    def _load_json_memories(self):
        """Load memories from JSON file."""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    for item in data.get("memories", []):
                        self._memories.append(LoopMemory.from_dict(item))
            except Exception as e:
                logger.error(f"Failed to load {self.loop_type} memories: {e}")
    
    def _save_json_memories(self):
        """Save memories to JSON file."""
        try:
            data = {
                "loop_type": self.loop_type.value,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "count": len(self._memories),
                "memories": [m.to_dict() for m in self._memories]
            }
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save {self.loop_type} memories: {e}")
    
    def _load_state(self):
        """Load loop state from database."""
        with self.db.cursor() as c:
            c.execute("SELECT state, parameters FROM loop_states WHERE loop_type = ?", 
                     (self.loop_type.value,))
            row = c.fetchone()
            if row:
                self.state = json.loads(row['state'])
                if row['parameters']:
                    self.state['parameters'] = json.loads(row['parameters'])
    
    def _save_state(self):
        """Save loop state to database."""
        with self.db.cursor() as c:
            c.execute("""
                INSERT OR REPLACE INTO loop_states (loop_type, state, last_updated, parameters)
                VALUES (?, ?, ?, ?)
            """, (
                self.loop_type.value,
                json.dumps(self.state),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(self.state.get('parameters', {}))
            ))
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new memory to this loop's system."""
        memory_id = hashlib.md5(
            f"{self.loop_type.value}:{datetime.now().isoformat()}:{content[:50]}".encode()
        ).hexdigest()[:16]
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        memory = LoopMemory(
            id=memory_id,
            timestamp=timestamp,
            content=content,
            metadata=metadata or {}
        )
        
        # Add to JSON hot memory
        self._memories.append(memory)
        self._save_json_memories()
        
        # Embed and add to vector index
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            vector = self.embedder.encode_single(content)
            
            # Store embedding in DB
            with self.db.cursor() as c:
                c.execute("""
                    INSERT INTO messages (timestamp, role, content, loop_type, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (timestamp, "loop", content, self.loop_type.value, json.dumps(metadata or {})))
                message_id = c.lastrowid
                
                c.execute("""
                    INSERT INTO embeddings (message_id, vector, created_at)
                    VALUES (?, ?, ?)
                """, (message_id, vector.tobytes(), timestamp))
                embedding_id = c.lastrowid
                
                # Update memory with embedding ID
                memory.embedding_id = embedding_id
            
            # Add to FAISS index
            self.vector_index.add(
                vector.reshape(1, -1),
                np.array([embedding_id], dtype=np.int64)
            )
        
        # Check if pruning needed
        if len(self._memories) >= JSON_PRUNE_THRESHOLD:
            self._prune_memories()
        
        # Update state
        self.state['processing_count'] += 1
        self.state['last_trigger'] = timestamp
        self._save_state()
        
        return memory_id
    
    def _prune_memories(self):
        """Prune old memories through compression pipeline."""
        if len(self._memories) < JSON_PRUNE_THRESHOLD:
            return
        
        # Get oldest memories to compress
        to_compress = list(self._memories)[:FAISS_COMPRESS_THRESHOLD]
        
        if not to_compress:
            return
        
        # Create summary of old memories
        texts = [m.content for m in to_compress]
        summary = self._generate_summary(texts)
        
        # Store summary in SQLite
        source_ids = [m.id for m in to_compress]
        with self.db.cursor() as c:
            c.execute("""
                INSERT INTO summaries (loop_type, summary, source_ids, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                self.loop_type.value,
                summary,
                json.dumps(source_ids),
                datetime.now(timezone.utc).isoformat()
            ))
        
        # Remove compressed memories from hot storage
        for _ in range(len(to_compress)):
            self._memories.popleft()
        
        self._save_json_memories()
        logger.info(f"Pruned {len(to_compress)} memories from {self.loop_type.value}")
    
    def _generate_summary(self, texts: List[str]) -> str:
        """Generate a summary of multiple texts."""
        # Simple extractive summary - take key sentences
        combined = " ".join(texts)
        sentences = combined.split('.')
        
        # Take first and last sentences plus longest middle one
        if len(sentences) <= 3:
            return combined[:500]
        
        key_sentences = [sentences[0]]
        if len(sentences) > 2:
            middle = sorted(sentences[1:-1], key=len, reverse=True)
            if middle:
                key_sentences.append(middle[0])
        key_sentences.append(sentences[-1])
        
        return '. '.join(s.strip() for s in key_sentences if s.strip())[:500]
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search memories semantically."""
        results = []
        
        # Search JSON hot memory first (simple text matching)
        query_lower = query.lower()
        for mem in reversed(list(self._memories)):  # Recent first
            if query_lower in mem.content.lower():
                results.append({
                    "id": mem.id,
                    "timestamp": mem.timestamp,
                    "content": mem.content,
                    "source": "json",
                    "score": 1.0
                })
                if len(results) >= k // 2:
                    break
        
        # Vector search for semantic matches
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.vector_index.total > 0:
            query_vec = self.embedder.encode_single(query)
            distances, ids = self.vector_index.search(query_vec, k)
            
            for dist, emb_id in zip(distances, ids):
                if emb_id < 0:
                    continue
                
                with self.db.cursor() as c:
                    c.execute("""
                        SELECT m.content, m.timestamp, m.metadata 
                        FROM messages m
                        JOIN embeddings e ON e.message_id = m.id
                        WHERE e.id = ? AND m.loop_type = ?
                    """, (int(emb_id), self.loop_type.value))
                    row = c.fetchone()
                    
                    if row:
                        results.append({
                            "id": str(emb_id),
                            "timestamp": row['timestamp'],
                            "content": row['content'],
                            "source": "vector",
                            "score": float(1.0 / (1.0 + dist))
                        })
        
        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for r in results:
            content_hash = hashlib.md5(r['content'].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(r)
        
        return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:k]
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent memories."""
        recent = list(self._memories)[-n:]
        return [
            {
                "id": m.id,
                "timestamp": m.timestamp,
                "content": m.content,
                "metadata": m.metadata
            }
            for m in reversed(recent)
        ]

# =============================================================================
# Self-Loop Processors
# =============================================================================

class SelfLoopProcessor:
    """Base class for self-loop processing."""
    
    def __init__(self, loop_type: LoopType, memory: LoopMemoryManager):
        self.loop_type = loop_type
        self.memory = memory
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return loop-specific output."""
        raise NotImplementedError
    
    def get_state(self) -> Dict[str, Any]:
        """Get current loop state."""
        return self.memory.state

class AffectiveLoop(SelfLoopProcessor):
    """Processes emotional valence and arousal."""
    
    POSITIVE_WORDS = {
        'love', 'great', 'good', 'excellent', 'happy', 'wonderful', 'thanks',
        'amazing', 'beautiful', 'joy', 'excited', 'fantastic', 'perfect',
        'appreciate', 'grateful', 'blessed', 'awesome', 'brilliant'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'error', 'failed', 'sad', 'terrible', 'hate', 'problem',
        'awful', 'horrible', 'angry', 'frustrated', 'disappointed', 'upset',
        'worry', 'fear', 'anxious', 'stress', 'wrong', 'broken'
    }
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text_lower = input_text.lower()
        words = set(re.findall(r'\w+', text_lower))
        
        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)
        total = positive_count + negative_count + 1
        
        valence = (positive_count - negative_count) / total
        arousal = min(1.0, (positive_count + negative_count) / 10)
        
        if valence > 0.1:
            state = "positive"
        elif valence < -0.1:
            state = "negative"
        else:
            state = "neutral"
        
        # Store affective reading
        self.memory.add_memory(
            f"Affective state: {state} (valence={valence:.2f}, arousal={arousal:.2f})",
            {"valence": valence, "arousal": arousal, "state": state}
        )
        
        return {
            "state": state,
            "valence": valence,
            "arousal": arousal,
            "dominant_affect": state
        }

class EmotionalLoop(SelfLoopProcessor):
    """Processes emotional understanding and empathy."""
    
    EMOTION_PATTERNS = {
        'joy': r'\b(happy|joy|excited|thrilled|delighted)\b',
        'sadness': r'\b(sad|unhappy|depressed|down|blue)\b',
        'anger': r'\b(angry|mad|furious|annoyed|irritated)\b',
        'fear': r'\b(afraid|scared|anxious|worried|nervous)\b',
        'surprise': r'\b(surprised|shocked|amazed|astonished)\b',
        'disgust': r'\b(disgusted|revolted|repulsed)\b',
        'trust': r'\b(trust|believe|faith|confident)\b',
        'anticipation': r'\b(expect|anticipate|hope|look forward)\b'
    }
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text_lower = input_text.lower()
        detected_emotions = {}
        
        for emotion, pattern in self.EMOTION_PATTERNS.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                detected_emotions[emotion] = len(matches)
        
        primary_emotion = max(detected_emotions, key=detected_emotions.get) if detected_emotions else "neutral"
        
        # Calculate empathy response
        empathy_response = self._generate_empathy(primary_emotion, detected_emotions)
        
        self.memory.add_memory(
            f"Emotional analysis: {primary_emotion} detected. {empathy_response}",
            {"emotions": detected_emotions, "primary": primary_emotion}
        )
        
        return {
            "emotions": detected_emotions,
            "primary_emotion": primary_emotion,
            "empathy_response": empathy_response
        }
    
    def _generate_empathy(self, emotion: str, all_emotions: Dict[str, int]) -> str:
        responses = {
            'joy': "I sense happiness in your words.",
            'sadness': "I understand this might be difficult.",
            'anger': "I can see you're frustrated.",
            'fear': "It's natural to feel uncertain.",
            'surprise': "That does sound unexpected.",
            'disgust': "I understand your reaction.",
            'trust': "I appreciate your confidence.",
            'anticipation': "Looking forward to good things.",
            'neutral': "I'm here to help."
        }
        return responses.get(emotion, responses['neutral'])

class CognitiveLoop(SelfLoopProcessor):
    """Processes reasoning and logical analysis."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Analyze query complexity
        word_count = len(input_text.split())
        sentence_count = len(re.split(r'[.!?]', input_text))
        question_count = input_text.count('?')
        
        complexity = "simple"
        if word_count > 50 or question_count > 2:
            complexity = "complex"
        elif word_count > 20 or question_count > 0:
            complexity = "moderate"
        
        # Identify cognitive task type
        task_type = self._identify_task(input_text)
        
        # Determine reasoning depth needed
        reasoning_depth = {
            "simple": 1,
            "moderate": 2,
            "complex": 3
        }[complexity]
        
        self.memory.add_memory(
            f"Cognitive analysis: {task_type} task, {complexity} complexity",
            {"task_type": task_type, "complexity": complexity, "depth": reasoning_depth}
        )
        
        return {
            "complexity": complexity,
            "task_type": task_type,
            "reasoning_depth": reasoning_depth,
            "word_count": word_count
        }
    
    def _identify_task(self, text: str) -> str:
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['explain', 'why', 'how does', 'what is']):
            return "explanation"
        elif any(w in text_lower for w in ['compare', 'difference', 'versus', 'vs']):
            return "comparison"
        elif any(w in text_lower for w in ['create', 'write', 'generate', 'make']):
            return "generation"
        elif any(w in text_lower for w in ['analyze', 'evaluate', 'assess']):
            return "analysis"
        elif any(w in text_lower for w in ['fix', 'debug', 'error', 'wrong']):
            return "troubleshooting"
        elif any(w in text_lower for w in ['summarize', 'summary', 'brief']):
            return "summarization"
        else:
            return "general"

class SelfReflectionLoop(SelfLoopProcessor):
    """Meta-cognitive reflection on responses."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a reflection trigger
        needs_reflection = any(
            trigger in input_text.lower()
            for trigger in ['think about', 'consider', 'reflect', 'reconsider', 'think again']
        )
        
        # Get previous responses to reflect on
        recent = self.memory.get_recent(3)
        
        reflection_notes = []
        if needs_reflection and recent:
            for mem in recent:
                reflection_notes.append(f"Previous thought: {mem['content'][:100]}")
        
        confidence = 0.8  # Base confidence
        if context.get('cognitive', {}).get('complexity') == 'complex':
            confidence -= 0.1
        if context.get('affective', {}).get('state') == 'negative':
            confidence -= 0.05
        
        self.memory.add_memory(
            f"Self-reflection: confidence={confidence:.2f}, needs_reflection={needs_reflection}",
            {"confidence": confidence, "reflection_triggered": needs_reflection}
        )
        
        return {
            "needs_reflection": needs_reflection,
            "confidence": confidence,
            "reflection_notes": reflection_notes
        }

class SelfLearningLoop(SelfLoopProcessor):
    """Extracts patterns and learnings from interactions."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Extract potential patterns
        patterns = []
        
        # Topic detection
        topics = self._extract_topics(input_text)
        if topics:
            patterns.append(f"Topics: {', '.join(topics)}")
        
        # User preference signals
        if 'prefer' in input_text.lower() or 'like' in input_text.lower():
            patterns.append("User expressing preference")
        
        if 'always' in input_text.lower() or 'never' in input_text.lower():
            patterns.append("User expressing strong preference")
        
        # Learning opportunity
        learning_opportunity = len(patterns) > 0
        
        if learning_opportunity:
            self.memory.add_memory(
                f"Learning: {'; '.join(patterns)}",
                {"patterns": patterns, "topics": topics}
            )
        
        return {
            "patterns_detected": patterns,
            "topics": topics,
            "learning_opportunity": learning_opportunity
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        # Simple noun extraction (would be better with NLP)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words
        return list(set(words))[:5]

class SelfAdjustmentLoop(SelfLoopProcessor):
    """Adjusts response parameters based on feedback."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        adjustments = {}
        
        # Check for feedback signals
        text_lower = input_text.lower()
        
        # Length adjustment
        if any(w in text_lower for w in ['shorter', 'brief', 'concise']):
            adjustments['max_tokens'] = 'decrease'
            adjustments['length_preference'] = 'short'
        elif any(w in text_lower for w in ['longer', 'detailed', 'elaborate', 'more']):
            adjustments['max_tokens'] = 'increase'
            adjustments['length_preference'] = 'long'
        
        # Style adjustment
        if any(w in text_lower for w in ['formal', 'professional']):
            adjustments['style'] = 'formal'
        elif any(w in text_lower for w in ['casual', 'friendly', 'relaxed']):
            adjustments['style'] = 'casual'
        
        # Temperature adjustment based on task
        task_type = context.get('cognitive', {}).get('task_type', 'general')
        if task_type in ['analysis', 'troubleshooting']:
            adjustments['temperature'] = 0.5
        elif task_type in ['generation', 'creative']:
            adjustments['temperature'] = 0.9
        else:
            adjustments['temperature'] = 0.7
        
        if adjustments:
            self.memory.add_memory(
                f"Adjustments: {json.dumps(adjustments)}",
                {"adjustments": adjustments}
            )
        
        return {
            "adjustments": adjustments,
            "temperature": adjustments.get('temperature', 0.7),
            "style": adjustments.get('style', 'balanced')
        }

class ContextualLoop(SelfLoopProcessor):
    """Manages conversation context and history."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Get relevant context from memory
        relevant_memories = self.memory.search(input_text, k=5)
        
        # Build context summary
        context_items = []
        for mem in relevant_memories:
            context_items.append({
                "content": mem['content'][:200],
                "timestamp": mem['timestamp'],
                "relevance": mem['score']
            })
        
        # Determine if this is a follow-up
        is_followup = any(
            w in input_text.lower()
            for w in ['also', 'and', 'another', 'what about', 'how about', 'continue']
        )
        
        self.memory.add_memory(
            f"Context: {len(relevant_memories)} relevant items, followup={is_followup}",
            {"memory_count": len(relevant_memories), "is_followup": is_followup}
        )
        
        return {
            "relevant_context": context_items,
            "is_followup": is_followup,
            "context_depth": len(relevant_memories)
        }

class TemporalLoop(SelfLoopProcessor):
    """Manages time awareness and temporal references."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        
        # Detect temporal references
        temporal_refs = []
        text_lower = input_text.lower()
        
        if 'today' in text_lower:
            temporal_refs.append(('today', now.date().isoformat()))
        if 'yesterday' in text_lower:
            from datetime import timedelta
            temporal_refs.append(('yesterday', (now - timedelta(days=1)).date().isoformat()))
        if 'tomorrow' in text_lower:
            from datetime import timedelta
            temporal_refs.append(('tomorrow', (now + timedelta(days=1)).date().isoformat()))
        if 'now' in text_lower or 'current' in text_lower:
            temporal_refs.append(('now', now.isoformat()))
        
        # Time-based context
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        self.memory.add_memory(
            f"Temporal: {time_of_day}, refs={temporal_refs}",
            {"time_of_day": time_of_day, "references": temporal_refs}
        )
        
        return {
            "current_time": now.isoformat(),
            "time_of_day": time_of_day,
            "temporal_references": temporal_refs,
            "day_of_week": now.strftime("%A")
        }

class SemanticLoop(SelfLoopProcessor):
    """Extracts semantic meaning and intent."""
    
    INTENT_PATTERNS = {
        'question': r'\?|^(what|who|where|when|why|how|is|are|can|could|would|will)\b',
        'command': r'^(do|make|create|write|show|tell|find|search|get)\b',
        'statement': r'\.$|^(i think|i believe|it is|there is)\b',
        'greeting': r'^(hi|hello|hey|good morning|good evening)\b',
        'farewell': r'\b(bye|goodbye|see you|later)\b',
        'gratitude': r'\b(thank|thanks|appreciate)\b',
        'apology': r'\b(sorry|apologize|my bad)\b'
    }
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text_lower = input_text.lower().strip()
        
        # Detect intent
        detected_intents = []
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_intents.append(intent)
        
        primary_intent = detected_intents[0] if detected_intents else 'statement'
        
        # Extract key entities (simple approach)
        entities = self._extract_entities(input_text)
        
        # Semantic density (information richness)
        word_count = len(input_text.split())
        unique_words = len(set(input_text.lower().split()))
        density = unique_words / max(word_count, 1)
        
        self.memory.add_memory(
            f"Semantic: intent={primary_intent}, entities={entities}",
            {"intent": primary_intent, "entities": entities, "density": density}
        )
        
        return {
            "primary_intent": primary_intent,
            "all_intents": detected_intents,
            "entities": entities,
            "semantic_density": density
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        # Extract quoted strings, capitalized words, and @mentions
        entities = []
        
        # Quoted strings
        entities.extend(re.findall(r'"([^"]+)"', text))
        entities.extend(re.findall(r"'([^']+)'", text))
        
        # Capitalized sequences
        entities.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text))
        
        # @mentions
        entities.extend(re.findall(r'@(\w+)', text))
        
        return list(set(entities))[:10]

class RelationalLoop(SelfLoopProcessor):
    """Models user relationship and preferences."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Build user profile from recent interactions
        recent_memories = self.memory.get_recent(20)
        
        # Aggregate patterns
        interaction_count = len(recent_memories)
        
        # Detect relationship signals
        text_lower = input_text.lower()
        rapport_signals = []
        
        if any(w in text_lower for w in ['we', 'us', 'our', 'together']):
            rapport_signals.append("collaborative")
        if any(w in text_lower for w in ['you', 'your']):
            rapport_signals.append("addressing")
        if any(w in text_lower for w in ['i', 'my', 'me']):
            rapport_signals.append("personal_sharing")
        
        # Trust level (simplified)
        trust_level = min(1.0, interaction_count / 50)
        
        self.memory.add_memory(
            f"Relational: trust={trust_level:.2f}, signals={rapport_signals}",
            {"trust_level": trust_level, "signals": rapport_signals}
        )
        
        return {
            "interaction_count": interaction_count,
            "trust_level": trust_level,
            "rapport_signals": rapport_signals,
            "relationship_stage": "established" if trust_level > 0.5 else "developing"
        }

class CreativeLoop(SelfLoopProcessor):
    """Handles creative generation and ideation."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text_lower = input_text.lower()
        
        # Detect creative task
        creative_triggers = {
            'write': ['write', 'compose', 'draft'],
            'imagine': ['imagine', 'envision', 'picture'],
            'create': ['create', 'make', 'design', 'build'],
            'story': ['story', 'narrative', 'tale'],
            'poem': ['poem', 'poetry', 'verse'],
            'idea': ['idea', 'brainstorm', 'suggest']
        }
        
        detected_type = None
        for ctype, triggers in creative_triggers.items():
            if any(t in text_lower for t in triggers):
                detected_type = ctype
                break
        
        # Creative mode settings
        if detected_type:
            creativity_level = 0.9
            freedom = "high"
        else:
            creativity_level = 0.5
            freedom = "moderate"
        
        self.memory.add_memory(
            f"Creative: type={detected_type}, level={creativity_level}",
            {"type": detected_type, "level": creativity_level}
        )
        
        return {
            "creative_type": detected_type,
            "creativity_level": creativity_level,
            "creative_freedom": freedom,
            "suggested_temperature": 0.9 if detected_type else 0.7
        }

class AnalyticalLoop(SelfLoopProcessor):
    """Handles analytical and data-focused tasks."""
    
    def process(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text_lower = input_text.lower()
        
        # Detect analytical task
        analytical_triggers = {
            'analyze': ['analyze', 'analysis', 'examine'],
            'compare': ['compare', 'contrast', 'difference'],
            'evaluate': ['evaluate', 'assess', 'judge'],
            'calculate': ['calculate', 'compute', 'math'],
            'statistics': ['statistics', 'data', 'numbers', 'percentage'],
            'logic': ['logic', 'reasoning', 'deduce', 'infer']
        }
        
        detected_type = None
        for atype, triggers in analytical_triggers.items():
            if any(t in text_lower for t in triggers):
                detected_type = atype
                break
        
        # Check for data presence
        has_numbers = bool(re.search(r'\d+', input_text))
        has_comparison = any(w in text_lower for w in ['vs', 'versus', 'than', 'better', 'worse'])
        
        precision_level = 0.5
        if detected_type or has_numbers:
            precision_level = 0.9
        
        self.memory.add_memory(
            f"Analytical: type={detected_type}, precision={precision_level}",
            {"type": detected_type, "precision": precision_level}
        )
        
        return {
            "analytical_type": detected_type,
            "has_data": has_numbers,
            "has_comparison": has_comparison,
            "precision_level": precision_level,
            "suggested_temperature": 0.3 if detected_type else 0.7
        }

# =============================================================================
# Agentic Tool System
# =============================================================================

class ToolRegistry:
    """Registry and executor for agentic tools."""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tools."""
        
        # Web search tool
        self.register(ToolDefinition(
            name="web_search",
            description="Search the web using DuckDuckGo",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            },
            handler=self._web_search,
            category="search"
        ))
        
        # File reader tool
        self.register(ToolDefinition(
            name="read_file",
            description="Read content from an uploaded file",
            parameters={
                "file_id": {"type": "integer", "description": "File ID from uploads"}
            },
            handler=self._read_file,
            category="files"
        ))
        
        # Memory search tool
        self.register(ToolDefinition(
            name="search_memory",
            description="Search through conversation memory",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "default": 5}
            },
            handler=self._search_memory,
            category="memory"
        ))
        
        # Calculator tool
        self.register(ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            handler=self._calculate,
            category="utility"
        ))
        
        # Current time tool
        self.register(ToolDefinition(
            name="current_time",
            description="Get the current date and time",
            parameters={},
            handler=self._current_time,
            category="utility"
        ))
        
        # Text-to-speech tool
        self.register(ToolDefinition(
            name="speak",
            description="Convert text to speech",
            parameters={
                "text": {"type": "string", "description": "Text to speak"},
                "voice": {"type": "string", "optional": True}
            },
            handler=self._speak,
            category="tts"
        ))
    
    def register(self, tool: ToolDefinition):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "category": t.category
            }
            for t in self.tools.values()
        ]
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any], context: Dict[str, Any] = None) -> ToolResult:
        """Execute a tool."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(success=False, output=None, error=f"Unknown tool: {tool_name}")
        
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(parameters, context or {})
            else:
                result = tool.handler(parameters, context or {})
            
            execution_time = time.time() - start_time
            return ToolResult(success=True, output=result, execution_time=execution_time)
        
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return ToolResult(success=False, output=None, error=str(e), execution_time=time.time() - start_time)
    
    # Built-in tool handlers
    def _web_search(self, params: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Execute web search."""
        if DDGS is None:
            return [{"error": "DuckDuckGo search not available"}]
        
        query = params.get("query", "")
        max_results = params.get("max_results", 5)
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "url": r.get("href", "")
                    }
                    for r in results
                ]
        except Exception as e:
            return [{"error": str(e)}]
    
    def _read_file(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Read uploaded file."""
        file_id = params.get("file_id")
        db = context.get("db")
        
        if not db:
            return {"error": "Database not available"}
        
        with db.cursor() as c:
            c.execute("SELECT filename, extracted_text FROM uploaded_files WHERE id = ?", (file_id,))
            row = c.fetchone()
            
            if row:
                return {
                    "filename": row['filename'],
                    "content": row['extracted_text'] or "No text extracted"
                }
            return {"error": "File not found"}
    
    def _search_memory(self, params: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search memory."""
        query = params.get("query", "")
        k = params.get("k", 5)
        memory_manager = context.get("contextual_memory")
        
        if memory_manager:
            return memory_manager.search(query, k)
        return []
    
    def _calculate(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Safe calculator."""
        expression = params.get("expression", "")
        
        # Sanitize expression
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters in expression"}
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _current_time(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Get current time."""
        now = datetime.now(timezone.utc)
        return {
            "utc": now.isoformat(),
            "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p UTC")
        }
    
    async def _speak(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Text to speech."""
        if edge_tts is None:
            return {"error": "Edge TTS not available"}
        
        text = params.get("text", "")
        voice = params.get("voice", "en-US-AriaNeural")
        
        try:
            filename = f"tts_{hashlib.md5(text.encode()).hexdigest()[:8]}.mp3"
            filepath = TTS_DIR / filename
            
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(filepath))
            
            return {
                "success": True,
                "file": str(filepath),
                "url": f"/tts/{filename}"
            }
        except Exception as e:
            return {"error": str(e)}

# =============================================================================
# Tool Planner (Agentic)
# =============================================================================

class ToolPlanner:
    """Plans tool usage based on query analysis."""
    
    TOOL_TRIGGERS = {
        "web_search": [
            # Trigger web search for general queries and when users ask for information
            # that likely requires the latest data or factual lookup.
            r'\b(search|find|look up|google|what is|who is|latest|news|current)\b',
            r'\b(online|internet|web)\b',
            # Add weather-related keywords to ensure weather queries invoke the web search.
            r'\b(weather|forecast|temperature|conditions)\b'
        ],
        "calculate": [
            r'\b(calculate|compute|math|add|subtract|multiply|divide)\b',
            r'\d+\s*[\+\-\*\/\%]\s*\d+'
        ],
        "current_time": [
            r'\b(time|date|today|now|what day)\b'
        ],
        "search_memory": [
            r'\b(remember|recall|previously|earlier|before|last time)\b'
        ],
        "read_file": [
            r'\b(file|document|uploaded|attachment)\b'
        ],
        "speak": [
            r'\b(say|speak|read aloud|voice|audio)\b'
        ]
    }
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def plan(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan which tools to use for a query."""
        query_lower = query.lower()
        planned_tools = []
        
        for tool_name, patterns in self.TOOL_TRIGGERS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    tool = self.registry.get(tool_name)
                    if tool:
                        params = self._extract_params(tool_name, query, context)
                        planned_tools.append({
                            "tool": tool_name,
                            "parameters": params,
                            "confidence": 0.8
                        })
                    break
        
        return planned_tools
    
    def _extract_params(self, tool_name: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for a tool from the query."""
        if tool_name == "web_search":
            # Use the query as search query
            return {"query": query, "max_results": 5}
        
        elif tool_name == "calculate":
            # Extract math expression
            match = re.search(r'[\d\+\-\*\/\%\(\)\.\s]+', query)
            if match:
                return {"expression": match.group().strip()}
            return {"expression": query}
        
        elif tool_name == "search_memory":
            return {"query": query, "k": 5}
        
        elif tool_name == "read_file":
            # Try to extract file ID
            match = re.search(r'file\s*#?(\d+)', query.lower())
            if match:
                return {"file_id": int(match.group(1))}
            return {}
        
        elif tool_name == "speak":
            return {"text": query}
        
        return {}

# =============================================================================
# LLM Adapter
# =============================================================================

class LLMAdapter:
    """Adapter for Ollama models."""
    
    def __init__(self, model: str = "qwen2:7b"):
        self.model = model
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Send chat request to Ollama."""
        if not OLLAMA_AVAILABLE:
            return "[Error: Ollama not available. Please install and start Ollama.]"
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response.get('message', {}).get('content', '')
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[Error: {e}]"
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Simple generation."""
        return self.chat([{"role": "user", "content": prompt}], temperature)

# =============================================================================
# File Processing
# =============================================================================

class FileProcessor:
    """Processes uploaded files."""
    
    # Supported file extensions. This set is intentionally expansive to cover
    # common text formats (.txt, .md), configuration/markup files (.json,
    # .yaml, .yml, .xml), source code across many languages (.py, .js, .ts,
    # .java, .c, etc.), as well as structured documents (.csv, .pdf, .docx,
    # .odt), spreadsheets (.xlsx) and presentation files (.pptx). Image
    # formats are also included to allow OCR when the necessary dependencies
    # are available. Unsupported or unhandled extensions will be ingested but
    # will not have any text extracted.
    SUPPORTED_EXTENSIONS = {
        # Plain text and markup
        '.txt', '.md', '.rst', '.cfg', '.conf', '.ini', '.toml', '.csv',
        '.json', '.yaml', '.yml', '.xml',

        # Source code (common languages)
        '.py', '.pyw', '.js', '.jsx', '.ts', '.tsx', '.c', '.h', '.cpp', '.hpp',
        '.cc', '.cxx', '.java', '.cs', '.go', '.rb', '.php', '.rs', '.swift',
        '.m', '.kt', '.kts', '.dart', '.sh', '.bash', '.zsh', '.ps1',
        '.sql', '.pl', '.pm', '.scala', '.hs', '.erl', '.ex', '.exs',
        '.clj', '.groovy', '.coffee', '.lua', '.nim', '.R', '.r', '.vb',
        '.html', '.htm', '.css', '.scss', '.less', '.sass', '.vue', '.svelte',
        '.tex',

        # Documents
        '.pdf', '.docx', '.odt', '.xlsx', '.pptx',

        # Images (for OCR)
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp'
    }
    
    @staticmethod
    def extract_text(filepath: Path, filename: str) -> Optional[str]:
        """Extract text from a file.

        This method attempts to read the given file and return its textual
        contents. It supports a wide variety of formats: plain text and
        configuration files, source code, PDF documents, Word documents,
        OpenDocument text, Excel spreadsheets, PowerPoint presentations and
        several image formats via OCR. When extraction fails or the format is
        unsupported, ``None`` is returned.

        Parameters
        ----------
        filepath : Path
            Absolute path to the file on disk.
        filename : str
            Original filename, used to determine the extension.

        Returns
        -------
        Optional[str]
            The extracted text, or ``None`` if extraction could not be
            performed or produced no output.
        """
        ext = Path(filename).suffix.lower()

        try:
            # Plain text and markup files: directly read the file contents.
            if ext in {
                '.txt', '.md', '.rst', '.cfg', '.conf', '.ini', '.toml', '.csv',
                '.json', '.yaml', '.yml', '.xml', '.html', '.htm', '.css',
                '.scss', '.less', '.sass', '.vue', '.svelte', '.tex'
            } | {
                # Source code files are treated as plain text
                '.py', '.pyw', '.js', '.jsx', '.ts', '.tsx', '.c', '.h', '.cpp', '.hpp',
                '.cc', '.cxx', '.java', '.cs', '.go', '.rb', '.php', '.rs', '.swift',
                '.m', '.kt', '.kts', '.dart', '.sh', '.bash', '.zsh', '.ps1',
                '.sql', '.pl', '.pm', '.scala', '.hs', '.erl', '.ex', '.exs',
                '.clj', '.groovy', '.coffee', '.lua', '.nim', '.r', '.vb'
            }:
                return filepath.read_text(encoding='utf-8', errors='ignore')

            # PDF documents
            if ext == '.pdf' and PyPDF2:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text_fragments: List[str] = []
                    for page in reader.pages:
                        # extract_text may return None for pages without text
                        extracted = page.extract_text() or ''
                        text_fragments.append(extracted)
                    return '\n'.join(text_fragments)

            # Word documents
            if ext == '.docx' and docx_module:
                try:
                    doc = docx_module.Document(str(filepath))
                    return '\n'.join(p.text for p in doc.paragraphs)
                except Exception as e:
                    logger.error(f"DOCX extraction failed for {filename}: {e}")
                    return None

            # OpenDocument text
            if ext == '.odt' and load_odf:
                try:
                    doc = load_odf(str(filepath))
                    paras = doc.getElementsByType(odf_text.P)
                    return '\n'.join(odf_teletype.extractText(p) for p in paras)
                except Exception as e:
                    logger.error(f"ODT extraction failed for {filename}: {e}")
                    return None

            # Excel spreadsheets
            if ext == '.xlsx' and openpyxl:
                try:
                    wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
                    texts: List[str] = []
                    for ws in wb.worksheets:
                        for row in ws.iter_rows(values_only=True):
                            # Flatten row values into a single line, separating cells by tab
                            line_parts = []
                            for cell in row:
                                if cell is None:
                                    continue
                                # Convert all cell values to strings
                                line_parts.append(str(cell))
                            # Only append a line if there is content
                            if line_parts:
                                texts.append('\t'.join(line_parts))
                        # Separate sheets with a blank line
                        texts.append('')
                    return '\n'.join(texts).strip() or None
                except Exception as e:
                    logger.error(f"XLSX extraction failed for {filename}: {e}")
                    return None

            # PowerPoint presentations
            if ext == '.pptx' and pptx_module:
                try:
                    # The Presentation class is accessible via the pptx module
                    prs = pptx_module.Presentation(str(filepath))
                    slide_texts: List[str] = []
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            # Many shapes expose a .text attribute when they contain text
                            if hasattr(shape, 'text'):
                                text = getattr(shape, 'text')
                                if text:
                                    slide_texts.append(text)
                    return '\n'.join(slide_texts) or None
                except Exception as e:
                    logger.error(f"PPTX extraction failed for {filename}: {e}")
                    return None

            # Image files via OCR
            if ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp'}:
                # Only attempt OCR if both PIL and pytesseract are available
                if PIL and pytesseract:
                    try:
                        image = PIL.Image.open(filepath)
                        # pytesseract.image_to_string returns the extracted text
                        text = pytesseract.image_to_string(image)
                        # strip to remove leading/trailing whitespace; return None if empty
                        return text.strip() or None
                    except Exception as e:
                        logger.error(f"OCR failed for {filename}: {e}")
                        return None
                # Without OCR libraries, return None
                return None

        except Exception as e:
            logger.error(f"Failed to extract text from {filename}: {e}")

        return None

# =============================================================================
# Edge TTS Manager
# =============================================================================

class TTSManager:
    """Manages Microsoft Edge TTS."""
    
    def __init__(self):
        self._voices: Optional[List[Dict[str, str]]] = None
    
    async def get_voices(self) -> List[Dict[str, str]]:
        """Get available Edge TTS voices."""
        if self._voices is not None:
            return self._voices
        
        if edge_tts is None:
            return []
        
        try:
            voices = await edge_tts.list_voices()
            self._voices = [
                {
                    "name": v["Name"],
                    "short_name": v["ShortName"],
                    "locale": v["Locale"],
                    "gender": v["Gender"]
                }
                for v in voices
            ]
            return self._voices
        except Exception as e:
            logger.error(f"Failed to get TTS voices: {e}")
            return []
    
    async def synthesize(
        self,
        text: str,
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> Optional[Path]:
        """Synthesize speech."""
        if edge_tts is None:
            return None
        
        try:
            filename = f"tts_{hashlib.md5(f'{text}{voice}{rate}'.encode()).hexdigest()[:12]}.mp3"
            filepath = TTS_DIR / filename
            
            # Check cache
            if filepath.exists():
                return filepath
            
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(str(filepath))
            
            return filepath
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

# =============================================================================
# Web Search Manager
# =============================================================================

class WebSearchManager:
    """Enhanced web search with ranking."""
    
    @staticmethod
    def _search_ddgs(query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Internal helper to query DuckDuckGo via the ddgs library. Returns an empty
        list if the library is unavailable or an error occurs.
        """
        if DDGS is None:
            return []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                entries: List[Dict[str, Any]] = []
                for i, r in enumerate(results):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    url = r.get("href", "")
                    # Simple relevance scoring based on query word overlap
                    query_words = set(query.lower().split())
                    title_matches = sum(1 for w in query_words if w in title.lower())
                    body_matches = sum(1 for w in query_words if w in body.lower())
                    relevance = (title_matches * 2 + body_matches) / max(len(query_words), 1)
                    entries.append({
                        "title": title,
                        "body": body,
                        "url": url,
                        "relevance": relevance,
                        "position": i + 1
                    })
                # Sort by relevance descending
                entries.sort(key=lambda x: x["relevance"], reverse=True)
                return entries
        except Exception as e:
            logger.error(f"DDGS search failed: {e}")
            return []

    @staticmethod
    def _search_duckduckgo_html(query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback web search using DuckDuckGo's HTML page. This method makes a
        direct HTTP request to DuckDuckGo and scrapes the top results. It
        performs only very basic parsing and may break if DuckDuckGo changes
        their markup. Returns a list of dictionaries with title, snippet,
        URL and relevance score.
        """
        import requests
        import re
        try:
            # Query the HTML version of DuckDuckGo search to avoid JavaScript
            params = {
                "q": query,
                "kl": "us-en",
                "t": "h_",
                "ia": "web"
            }
            resp = requests.get("https://duckduckgo.com/html/", params=params, timeout=10)
            html = resp.text
            # Find result blocks. DuckDuckGo wraps each result in a div with class "result".
            # Extract the link and title using regex. This is a simple parser and may need
            # adjustments if DuckDuckGo alters its HTML.
            pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE)
            matches = pattern.findall(html)
            results: List[Dict[str, Any]] = []
            # Basic snippet extraction: the snippet often appears in a <a class="result__snippet"> tag
            snippet_pattern = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.IGNORECASE)
            snippets = [re.sub('<.*?>', '', m) for m in snippet_pattern.findall(html)]
            for i, (url, title) in enumerate(matches[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                # Compute relevance using simple word overlap
                query_words = set(query.lower().split())
                title_lower = re.sub('<.*?>', '', title).lower()
                snippet_lower = snippet.lower()
                title_matches = sum(1 for w in query_words if w in title_lower)
                snippet_matches = sum(1 for w in query_words if w in snippet_lower)
                relevance = (title_matches * 2 + snippet_matches) / max(len(query_words), 1)
                results.append({
                    "title": re.sub('<.*?>', '', title),
                    "body": snippet,
                    "url": url,
                    "relevance": relevance,
                    "position": i + 1
                })
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []

    @staticmethod
    def search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web for the given query and return ranked results. This
        method first tries to use the ddgs library (if available) for a clean
        JSON API. If that fails or returns no results, it falls back to
        scraping DuckDuckGo's HTML. Results are ranked based on a simple
        relevance score.
        """
        # Try the ddgs-based search first
        results = WebSearchManager._search_ddgs(query, max_results)
        if results:
            return results
        # Fallback to HTML scraping
        return WebSearchManager._search_duckduckgo_html(query, max_results)
    
    @staticmethod
    def search_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search news."""
        if DDGS is None:
            return []
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "url": r.get("url", ""),
                        "source": r.get("source", ""),
                        "date": r.get("date", "")
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []

# =============================================================================
# Main GPC3 Engine
# =============================================================================

class GPC3Engine:
    """Main GPC3 processing engine."""
    
    def __init__(self):
        # Initialize database
        self.db = DatabaseManager(DB_PATH)
        
        # Initialize embedding engine
        self.embedder = EmbeddingEngine(EMBED_MODEL_NAME)
        
        # Initialize vector index
        self.vector_index = VectorIndex(self.embedder.dimension, FAISS_INDEX_PATH)
        
        # Initialize all self-loops
        self.loops: Dict[LoopType, SelfLoopProcessor] = {}
        self.loop_memories: Dict[LoopType, LoopMemoryManager] = {}
        self._init_loops()
        
        # Initialize tool system
        self.tool_registry = ToolRegistry()
        self.tool_planner = ToolPlanner(self.tool_registry)
        
        # Initialize TTS
        self.tts = TTSManager()
        
        # Initialize web search
        self.web_search = WebSearchManager()
        
        # Load config
        self.config = self._load_config()
        
        # LLM adapter
        self.llm = LLMAdapter(self.config.get("default_model", "qwen2:7b"))
        
        logger.info("GPC3 Engine initialized")
    
    def _init_loops(self):
        """Initialize all cognitive self-loops."""
        loop_classes = {
            LoopType.AFFECTIVE: AffectiveLoop,
            LoopType.EMOTIONAL: EmotionalLoop,
            LoopType.COGNITIVE: CognitiveLoop,
            LoopType.SELF_REFLECTION: SelfReflectionLoop,
            LoopType.SELF_LEARNING: SelfLearningLoop,
            LoopType.SELF_ADJUSTMENT: SelfAdjustmentLoop,
            LoopType.CONTEXTUAL: ContextualLoop,
            LoopType.TEMPORAL: TemporalLoop,
            LoopType.SEMANTIC: SemanticLoop,
            LoopType.RELATIONAL: RelationalLoop,
            LoopType.CREATIVE: CreativeLoop,
            LoopType.ANALYTICAL: AnalyticalLoop
        }
        
        for loop_type, loop_class in loop_classes.items():
            memory = LoopMemoryManager(loop_type, self.db, self.embedder, self.vector_index)
            self.loop_memories[loop_type] = memory
            self.loops[loop_type] = loop_class(loop_type, memory)
            logger.info(f"Initialized loop: {loop_type.value}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        default_config = {
            "default_model": "qwen2:7b",
            "reflection_depth": 1,
            "max_memory": 100,
            "modes": {
                "default": {"temperature": 0.7, "use_web": False, "use_memory": True},
                "creative": {"temperature": 0.9, "use_web": True, "use_memory": True},
                "precise": {"temperature": 0.3, "use_web": False, "use_memory": True}
            }
        }
        
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r') as f:
                    loaded = json.load(f)
                    return {**default_config, **loaded}
            except Exception:
                pass
        
        return default_config
    
    def _save_config(self):
        """Save configuration."""
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message through all systems."""
        user_text = request.message.strip()
        
        if not user_text:
            return ChatResponse(answer="Please provide a message.")
        
        # Run all loops to build context
        loop_context = {}
        for loop_type, processor in self.loops.items():
            try:
                loop_context[loop_type.value] = processor.process(user_text, loop_context)
            except Exception as e:
                logger.error(f"Loop {loop_type.value} failed: {e}")
                loop_context[loop_type.value] = {}
        
        # Determine parameters from loop outputs
        affective_state = loop_context.get('affective', {}).get('state', 'neutral')
        temperature = loop_context.get('self_adjustment', {}).get('temperature', 0.7)
        
        # Plan and execute tools if enabled
        tools_used = []
        tool_results = []
        
        if request.use_tools:
            planned = self.tool_planner.plan(user_text, loop_context)
            
            for plan in planned:
                context = {
                    "db": self.db,
                    "contextual_memory": self.loop_memories[LoopType.CONTEXTUAL]
                }
                result = await self.tool_registry.execute(
                    plan['tool'],
                    plan['parameters'],
                    context
                )
                
                if result.success:
                    tools_used.append(plan['tool'])
                    tool_results.append({
                        "tool": plan['tool'],
                        "output": result.output
                    })
        
        # Get memory context if enabled
        memory_chunks = []
        if request.use_memory:
            memory_chunks = self.loop_memories[LoopType.CONTEXTUAL].search(user_text, k=5)
        
        # Get web context if enabled
        web_snippets = []
        if request.use_web:
            web_snippets = self.web_search.search(user_text, max_results=3)
        
        # Build prompt with all context
        system_prompt = self._build_system_prompt(loop_context)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add memory context
        for mem in memory_chunks:
            messages.append({
                "role": "assistant",
                "content": f"[Memory from {mem['timestamp']}] {mem['content']}"
            })
        
        # Add web context
        for snippet in web_snippets:
            messages.append({
                "role": "assistant",
                "content": f"[Web: {snippet['title']}] {snippet['body']}"
            })
        
        # Add tool results
        for tr in tool_results:
            messages.append({
                "role": "assistant",
                "content": f"[Tool {tr['tool']}] {json.dumps(tr['output'])}"
            })
        
        # Add user message
        messages.append({"role": "user", "content": user_text})
        
        # Select model
        model_name = request.model or self.config.get("default_model", "qwen2:7b")
        adapter = LLMAdapter(model_name)
        
        # Generate response
        answer = adapter.chat(messages, temperature=temperature)
        
        # Run reflection if configured
        reflection_depth = self.config.get("reflection_depth", 0)
        if reflection_depth > 0:
            for _ in range(reflection_depth):
                improved = self._reflect_on_answer(user_text, answer, messages, adapter, temperature)
                if improved and improved != answer:
                    answer = improved
                else:
                    break
        
        # Store interaction in memories
        self.loop_memories[LoopType.CONTEXTUAL].add_memory(
            f"User: {user_text}\nAssistant: {answer[:500]}",
            {"type": "conversation"}
        )
        
        # Store in database
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.db.cursor() as c:
            c.execute(
                "INSERT INTO messages (timestamp, role, content) VALUES (?, ?, ?)",
                (timestamp, "user", user_text)
            )
            c.execute(
                "INSERT INTO messages (timestamp, role, content) VALUES (?, ?, ?)",
                (timestamp, "assistant", answer)
            )
        
        return ChatResponse(
            answer=answer,
            memories=memory_chunks,
            web_snippets=web_snippets,
            affective=affective_state,
            loop_states={k: v for k, v in loop_context.items()},
            tools_used=tools_used if tools_used else None
        )
    
    def _build_system_prompt(self, loop_context: Dict[str, Any]) -> str:
        """Build system prompt from loop context."""
        parts = [
            "You are a sophisticated AI assistant with multiple cognitive systems.",
            "Respond naturally while leveraging the context and memories provided.",
            "Be helpful, accurate, and maintain emotional awareness.",
            # Instruct the model to leverage available tools when necessary.
            "You have access to external tools that can fetch up‑to‑date information (e.g., 'web_search' for retrieving current or unknown facts, 'current_time' for the exact date and time, 'calculate' for computations, 'read_file' for uploaded document contents, and 'speak' for text‑to‑speech).",
            "When a question requires information that may have changed after your training data, or when you are unsure of the answer, invoke the appropriate tool rather than guessing.",
            "Only answer from memory when no tool is relevant. After invoking a tool, incorporate its results into your final response. Do not fabricate real‑time data; always rely on tool output for factual answers."
        ]
        
        # Add loop insights
        emotional = loop_context.get('emotional', {})
        if emotional.get('empathy_response'):
            parts.append(f"Note: {emotional['empathy_response']}")
        
        cognitive = loop_context.get('cognitive', {})
        if cognitive.get('task_type'):
            parts.append(f"This appears to be a {cognitive['task_type']} task.")
        
        temporal = loop_context.get('temporal', {})
        if temporal.get('time_of_day'):
            parts.append(f"Current time context: {temporal['time_of_day']}")
        
        adjustment = loop_context.get('self_adjustment', {})
        if adjustment.get('style'):
            parts.append(f"Preferred style: {adjustment['style']}")
        
        return " ".join(parts)
    
    def _reflect_on_answer(
        self,
        user_text: str,
        raw_answer: str,
        messages: List[Dict[str, str]],
        adapter: LLMAdapter,
        temperature: float
    ) -> Optional[str]:
        """Perform self-reflection on generated answer."""
        critic_prompt = f"""You are critiquing your own response.

QUESTION: {user_text}

YOUR ANSWER: {raw_answer}

Task:
1. Identify any errors or areas for improvement
2. Provide an improved answer if needed

Respond with:
CRITIQUE: [your critique]
IMPROVED_ANSWER: [improved response or "NO CHANGE" if the original is good]"""

        try:
            response = adapter.chat([
                {"role": "system", "content": "You are a thoughtful critic."},
                {"role": "user", "content": critic_prompt}
            ], temperature=max(0.3, temperature - 0.2))
            
            # Parse improved answer
            if "IMPROVED_ANSWER:" in response:
                improved = response.split("IMPROVED_ANSWER:")[-1].strip()
                if improved and "NO CHANGE" not in improved.upper():
                    return improved
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
        
        return None
    
    async def upload_file(self, file: UploadFile) -> Dict[str, Any]:
        """Process an uploaded file."""
        filename = file.filename or "unknown"
        content = await file.read()
        content_hash = hashlib.md5(content).hexdigest()
        
        # Save file
        filepath = UPLOADS_DIR / f"{content_hash}_{filename}"
        with open(filepath, 'wb') as f:
            f.write(content)
        
        # Extract text
        extracted_text = FileProcessor.extract_text(filepath, filename)
        
        # Store in database
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.db.cursor() as c:
            c.execute("""
                INSERT INTO uploaded_files (filename, filepath, content_hash, extracted_text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, str(filepath), content_hash, extracted_text, timestamp))
            file_id = c.lastrowid
        
        # Embed and store in memory if text extracted
        fragments = 0
        if extracted_text:
            # Split into chunks
            chunks = self._chunk_text(extracted_text)
            for chunk in chunks:
                self.loop_memories[LoopType.CONTEXTUAL].add_memory(
                    f"[File: {filename}] {chunk}",
                    {"source": "upload", "file_id": file_id}
                )
                fragments += 1
            
            # Mark as embedded
            with self.db.cursor() as c:
                c.execute("UPDATE uploaded_files SET embedded = 1 WHERE id = ?", (file_id,))
        
        return {
            "file_id": file_id,
            "filename": filename,
            "fragments": fragments,
            "text_extracted": extracted_text is not None
        }
    
    def _chunk_text(self, text: str, max_chunk: int = 500) -> List[str]:
        """Split text into chunks for embedding."""
        paragraphs = text.split('\n\n')
        chunks = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_len + len(para) > max_chunk and current:
                chunks.append('\n'.join(current))
                current = []
                current_len = 0
            
            current.append(para)
            current_len += len(para)
        
        if current:
            chunks.append('\n'.join(current))
        
        return chunks

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="GPC3 API",
    description="Ghost Protocol Consciousness 3 - AI Backend",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize engine
engine: Optional[GPC3Engine] = None

@app.on_event("startup")
async def startup():
    global engine
    engine = GPC3Engine()
    logger.info("GPC3 API started")

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "online",
        "name": "GPC3",
        "version": "3.0.0",
        "loops_active": len(engine.loops) if engine else 0
    }

@app.get("/models")
async def get_models():
    """Get available Ollama models."""
    models = []
    
    if OLLAMA_AVAILABLE:
        try:
            data = ollama.list()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
    
    # Fallback to CLI
    if not models:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            for line in result.stdout.splitlines()[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
        except Exception:
            pass
    
    return {"models": models}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return await engine.process_message(request)

@app.post("/files/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process files."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    results = {}
    for file in files:
        result = await engine.upload_file(file)
        results[file.filename] = result
    
    return {"ingested": results}

@app.get("/files")
async def list_files():
    """List uploaded files."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    files = []
    with engine.db.cursor() as c:
        c.execute("SELECT id, filename, created_at, embedded FROM uploaded_files ORDER BY created_at DESC")
        for row in c.fetchall():
            files.append({
                "id": row['id'],
                "filename": row['filename'],
                "created_at": row['created_at'],
                "embedded": bool(row['embedded'])
            })
    
    return {"files": files}

@app.post("/memory/search")
async def search_memory(query: str, k: int = 5):
    """Search memory."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    results = engine.loop_memories[LoopType.CONTEXTUAL].search(query, k)
    return {"results": results}

@app.post("/web/search")
async def web_search(query: str, max_results: int = 5):
    """Web search."""
    results = WebSearchManager.search(query, max_results)
    return {"results": results}

@app.get("/tools")
async def list_tools():
    """List available tools."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {"tools": engine.tool_registry.list_tools()}

@app.post("/tools/execute")
async def execute_tool(request: ToolExecuteRequest):
    """Execute a tool."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    context = {
        "db": engine.db,
        "contextual_memory": engine.loop_memories[LoopType.CONTEXTUAL]
    }
    
    result = await engine.tool_registry.execute(
        request.tool_name,
        request.parameters,
        context
    )
    
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time": result.execution_time
    }

@app.get("/tts/voices")
async def get_tts_voices():
    """Get available TTS voices."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    voices = await engine.tts.get_voices()
    return {"voices": voices}

@app.post("/tts/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    filepath = await engine.tts.synthesize(
        request.text,
        request.voice or "en-US-AriaNeural",
        request.rate or "+0%",
        request.pitch or "+0Hz"
    )
    
    if filepath:
        return {"url": f"/tts/{filepath.name}"}
    
    raise HTTPException(status_code=500, detail="TTS synthesis failed")

@app.get("/tts/{filename}")
async def get_tts_file(filename: str):
    """Serve TTS audio file."""
    filepath = TTS_DIR / filename
    if filepath.exists():
        return FileResponse(filepath, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/loops")
async def get_loop_states():
    """Get states of all self-loops."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    states = {}
    for loop_type, processor in engine.loops.items():
        states[loop_type.value] = {
            "state": processor.get_state(),
            "recent_memories": engine.loop_memories[loop_type].get_recent(3)
        }
    
    return {"loops": states}

@app.get("/loops/{loop_type}")
async def get_loop_detail(loop_type: str):
    """Get detailed state of a specific loop."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        lt = LoopType(loop_type)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown loop type: {loop_type}")
    
    processor = engine.loops[lt]
    memory = engine.loop_memories[lt]
    
    return {
        "loop_type": loop_type,
        "state": processor.get_state(),
        "recent_memories": memory.get_recent(10),
        "memory_count": len(memory._memories)
    }

@app.get("/config")
async def get_config():
    """Get current configuration."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return engine.config

@app.post("/config")
async def update_config(new_config: Dict[str, Any]):
    """Update configuration."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    engine.config.update(new_config)
    engine._save_config()
    
    # Update LLM if model changed
    if "default_model" in new_config:
        engine.llm = LLMAdapter(new_config["default_model"])
    
    return {"status": "updated", "config": engine.config}

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    stats = {
        "loops": len(engine.loops),
        "tools": len(engine.tool_registry.tools),
        "vector_index_size": engine.vector_index.total,
        "features": {
            "embeddings": SENTENCE_TRANSFORMERS_AVAILABLE,
            "faiss": FAISS_AVAILABLE,
            "ollama": OLLAMA_AVAILABLE,
            "web_search": DDGS is not None,
            "tts": edge_tts is not None
        }
    }
    
    # Memory stats per loop
    stats["loop_memories"] = {}
    for lt, memory in engine.loop_memories.items():
        stats["loop_memories"][lt.value] = len(memory._memories)
    
    # Database stats
    with engine.db.cursor() as c:
        c.execute("SELECT COUNT(*) FROM messages")
        stats["total_messages"] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM uploaded_files")
        stats["total_files"] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM summaries")
        stats["total_summaries"] = c.fetchone()[0]
    
    return stats

# =============================================================================
# Enhanced Web Search with RAG
# =============================================================================

class WebRAGManager:
    """Web-based Retrieval Augmented Generation."""
    
    def __init__(self, embedder: EmbeddingEngine, vector_index: VectorIndex, db: DatabaseManager):
        self.embedder = embedder
        self.vector_index = vector_index
        self.db = db
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def search_and_embed(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search web and embed results for RAG."""
        # Check cache
        cache_key = hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['results']
        
        # Perform search
        results = WebSearchManager.search(query, max_results)
        
        if not results:
            return []
        
        # Embed and store results
        enhanced_results = []
        for r in results:
            text = f"{r['title']}. {r['body']}"
            
            # Embed
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                vector = self.embedder.encode_single(text)
                
                # Store in database
                timestamp = datetime.now(timezone.utc).isoformat()
                with self.db.cursor() as c:
                    c.execute("""
                        INSERT INTO messages (timestamp, role, content, loop_type, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (timestamp, "web", text, "web_rag", json.dumps({
                        "url": r['url'],
                        "query": query
                    })))
                    msg_id = c.lastrowid
                    
                    c.execute("""
                        INSERT INTO embeddings (message_id, vector, created_at)
                        VALUES (?, ?, ?)
                    """, (msg_id, vector.tobytes(), timestamp))
                    emb_id = c.lastrowid
                
                # Add to vector index
                self.vector_index.add(
                    vector.reshape(1, -1),
                    np.array([emb_id], dtype=np.int64)
                )
                
                r['embedding_id'] = emb_id
            
            enhanced_results.append(r)
        
        # Cache results
        self.cache[cache_key] = {
            'results': enhanced_results,
            'timestamp': time.time()
        }
        
        return enhanced_results
    
    def semantic_rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank results using semantic similarity."""
        if not results or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return results[:top_k]
        
        query_vec = self.embedder.encode_single(query)
        
        scored_results = []
        for r in results:
            text = f"{r['title']}. {r['body']}"
            text_vec = self.embedder.encode_single(text)
            
            # Cosine similarity
            similarity = np.dot(query_vec, text_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(text_vec) + 1e-8
            )
            
            scored_results.append({
                **r,
                'semantic_score': float(similarity)
            })
        
        # Sort by semantic score
        scored_results.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        return scored_results[:top_k]

# =============================================================================
# Streaming Response Handler
# =============================================================================

class StreamingHandler:
    """Handles streaming responses from LLM."""
    
    def __init__(self, model: str = "qwen2:7b"):
        self.model = model
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ):
        """Stream chat response tokens."""
        if not OLLAMA_AVAILABLE:
            yield {"error": "Ollama not available"}
            return
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature}
            )
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield {
                        "token": chunk['message']['content'],
                        "done": chunk.get('done', False)
                    }
        
        except Exception as e:
            yield {"error": str(e)}

# =============================================================================
# Conversation Manager
# =============================================================================

class ConversationManager:
    """Manages conversation history and context windows."""
    
    def __init__(self, db: DatabaseManager, max_history: int = 50):
        self.db = db
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
    
    def get_or_create_conversation(self, conversation_id: str = None) -> str:
        """Get or create a conversation."""
        if conversation_id and conversation_id in self.conversations:
            return conversation_id
        
        new_id = hashlib.md5(
            f"{datetime.now().isoformat()}:{id(self)}".encode()
        ).hexdigest()[:16]
        
        self.conversations[new_id] = []
        return new_id
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Trim if too long
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = \
                self.conversations[conversation_id][-self.max_history:]
    
    def get_history(self, conversation_id: str, n: int = None) -> List[Dict[str, str]]:
        """Get conversation history."""
        history = self.conversations.get(conversation_id, [])
        if n:
            return history[-n:]
        return history
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_context_window(
        self,
        conversation_id: str,
        max_tokens: int = 4000
    ) -> List[Dict[str, str]]:
        """Get messages that fit within token limit."""
        history = self.get_history(conversation_id)
        
        # Simple token estimation: 4 chars ≈ 1 token
        result = []
        total_chars = 0
        max_chars = max_tokens * 4
        
        for msg in reversed(history):
            msg_chars = len(msg['content'])
            if total_chars + msg_chars > max_chars:
                break
            result.insert(0, {"role": msg['role'], "content": msg['content']})
            total_chars += msg_chars
        
        return result

# =============================================================================
# Advanced Memory Compression
# =============================================================================

class MemoryCompressor:
    """Advanced memory compression using LLM summarization."""
    
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
    
    def compress_memories(
        self,
        memories: List[Dict[str, Any]],
        compression_ratio: float = 0.3
    ) -> str:
        """Compress multiple memories into a summary."""
        if not memories:
            return ""
        
        # Build context from memories
        context_parts = []
        for i, mem in enumerate(memories):
            timestamp = mem.get('timestamp', 'unknown')
            content = mem.get('content', '')[:500]
            context_parts.append(f"[{i+1}] ({timestamp}): {content}")
        
        context = "\n".join(context_parts)
        target_length = int(len(context) * compression_ratio)
        
        prompt = f"""Compress the following memories into a concise summary.
Target length: approximately {target_length} characters.
Preserve key facts, decisions, and important context.

MEMORIES:
{context}

COMPRESSED SUMMARY:"""

        try:
            summary = self.llm.chat([
                {"role": "system", "content": "You are a memory compression assistant. Be concise but preserve important information."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            return summary.strip()
        except Exception as e:
            logger.error(f"Memory compression failed: {e}")
            # Fallback: extractive summary
            return self._extractive_summary(memories, target_length)
    
    def _extractive_summary(self, memories: List[Dict[str, Any]], max_length: int) -> str:
        """Simple extractive summary fallback."""
        texts = [m.get('content', '')[:100] for m in memories]
        combined = " | ".join(texts)
        return combined[:max_length]

# =============================================================================
# Sentiment Analysis Engine
# =============================================================================

class SentimentAnalyzer:
    """Advanced sentiment analysis."""
    
    EMOTION_LEXICON = {
        'joy': {
            'words': ['happy', 'joy', 'delighted', 'pleased', 'excited', 'thrilled', 'elated', 'ecstatic'],
            'weight': 1.0
        },
        'sadness': {
            'words': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'heartbroken', 'gloomy'],
            'weight': -1.0
        },
        'anger': {
            'words': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'enraged', 'outraged'],
            'weight': -0.8
        },
        'fear': {
            'words': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'fearful'],
            'weight': -0.6
        },
        'surprise': {
            'words': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'startled'],
            'weight': 0.3
        },
        'trust': {
            'words': ['trust', 'confident', 'believe', 'faith', 'reliable', 'dependable'],
            'weight': 0.7
        },
        'anticipation': {
            'words': ['expect', 'anticipate', 'await', 'hope', 'looking forward'],
            'weight': 0.5
        },
        'disgust': {
            'words': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            'weight': -0.9
        }
    }
    
    INTENSIFIERS = {
        'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'incredibly': 1.7,
        'absolutely': 1.6, 'totally': 1.4, 'completely': 1.5, 'utterly': 1.6
    }
    
    NEGATORS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis."""
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        # Detect emotions
        emotion_scores = {}
        for emotion, data in self.EMOTION_LEXICON.items():
            score = 0
            for word in data['words']:
                if word in text_lower:
                    # Check for intensifiers and negators
                    word_idx = text_lower.find(word)
                    context = text_lower[max(0, word_idx-20):word_idx]
                    
                    multiplier = 1.0
                    for intensifier, value in self.INTENSIFIERS.items():
                        if intensifier in context:
                            multiplier *= value
                    
                    negated = any(neg in context for neg in self.NEGATORS)
                    if negated:
                        multiplier *= -0.5
                    
                    score += data['weight'] * multiplier
            
            if score != 0:
                emotion_scores[emotion] = score
        
        # Calculate overall sentiment
        if emotion_scores:
            total_score = sum(emotion_scores.values())
            if total_score > 0.3:
                overall = 'positive'
            elif total_score < -0.3:
                overall = 'negative'
            else:
                overall = 'neutral'
            
            primary_emotion = max(emotion_scores, key=lambda k: abs(emotion_scores[k]))
        else:
            overall = 'neutral'
            primary_emotion = 'neutral'
            total_score = 0
        
        # Calculate confidence
        confidence = min(1.0, abs(total_score) / 3)
        
        return {
            'overall': overall,
            'score': total_score,
            'confidence': confidence,
            'primary_emotion': primary_emotion,
            'emotions': emotion_scores,
            'word_count': len(words)
        }

# =============================================================================
# Intent Classifier
# =============================================================================

class IntentClassifier:
    """Classifies user intent."""
    
    INTENT_PATTERNS = {
        'greeting': {
            'patterns': [r'^(hi|hello|hey|greetings|good morning|good evening|good afternoon)\b'],
            'response_type': 'greeting'
        },
        'farewell': {
            'patterns': [r'\b(bye|goodbye|see you|later|farewell|goodnight)\b'],
            'response_type': 'farewell'
        },
        'question_factual': {
            'patterns': [r'^(what|who|where|when|which)\b.*\?'],
            'response_type': 'informational'
        },
        'question_how': {
            'patterns': [r'^how\b.*\?'],
            'response_type': 'instructional'
        },
        'question_why': {
            'patterns': [r'^why\b.*\?'],
            'response_type': 'explanatory'
        },
        'command': {
            'patterns': [r'^(do|make|create|write|show|find|search|get|calculate)\b'],
            'response_type': 'action'
        },
        'request': {
            'patterns': [r'\b(please|could you|would you|can you|help me)\b'],
            'response_type': 'assistance'
        },
        'opinion': {
            'patterns': [r'\b(think|believe|feel|opinion|thoughts)\b'],
            'response_type': 'conversational'
        },
        'comparison': {
            'patterns': [r'\b(compare|difference|better|worse|versus|vs)\b'],
            'response_type': 'analytical'
        },
        'definition': {
            'patterns': [r'\b(define|meaning|what is|what are)\b'],
            'response_type': 'definitional'
        },
        'creative': {
            'patterns': [r'\b(write|compose|create|imagine|story|poem)\b'],
            'response_type': 'creative'
        },
        'code': {
            'patterns': [r'\b(code|program|function|script|debug|error)\b'],
            'response_type': 'technical'
        },
        'summary': {
            'patterns': [r'\b(summarize|summary|brief|tldr|overview)\b'],
            'response_type': 'summary'
        },
        'feedback': {
            'patterns': [r'\b(thanks|thank you|great|awesome|good job|well done)\b'],
            'response_type': 'acknowledgment'
        }
    }
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify user intent."""
        text_lower = text.lower().strip()
        
        detected_intents = []
        
        for intent_name, data in self.INTENT_PATTERNS.items():
            for pattern in data['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_intents.append({
                        'intent': intent_name,
                        'response_type': data['response_type'],
                        'confidence': 0.8
                    })
                    break
        
        if not detected_intents:
            # Default classification based on punctuation
            if text.strip().endswith('?'):
                detected_intents.append({
                    'intent': 'question_general',
                    'response_type': 'informational',
                    'confidence': 0.5
                })
            else:
                detected_intents.append({
                    'intent': 'statement',
                    'response_type': 'conversational',
                    'confidence': 0.5
                })
        
        primary = detected_intents[0]
        
        return {
            'primary_intent': primary['intent'],
            'response_type': primary['response_type'],
            'confidence': primary['confidence'],
            'all_intents': detected_intents
        }

# =============================================================================
# Topic Extractor
# =============================================================================

class TopicExtractor:
    """Extracts topics and entities from text."""
    
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
        'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
        'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
    }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract topics and entities."""
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        content_words = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
        
        # Count word frequencies
        word_freq = {}
        for word in content_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Extract proper nouns (capitalized words)
        proper_nouns = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)))
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
        
        # Extract email-like patterns
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        
        # Extract numbers with context
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|k|m|b|million|billion|thousand)?\b', text.lower())
        
        return {
            'topics': [{'word': w, 'frequency': f} for w, f in topics],
            'entities': proper_nouns[:10],
            'quoted_strings': quoted,
            'urls': urls,
            'emails': emails,
            'numbers': numbers,
            'word_count': len(words),
            'unique_words': len(set(words))
        }

# =============================================================================
# Response Generator
# =============================================================================

class ResponseGenerator:
    """Generates contextual responses."""
    
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
    
    def generate_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate a response with full context."""
        # Build comprehensive prompt
        system_parts = ["You are gpc3, an advanced AI assistant with multiple cognitive systems."]
        
        # Add emotional context
        if 'sentiment' in context:
            sentiment = context['sentiment']
            if sentiment['overall'] == 'negative':
                system_parts.append("The user seems distressed. Be empathetic and supportive.")
            elif sentiment['overall'] == 'positive':
                system_parts.append("The user seems positive. Match their energy.")
        
        # Add intent context
        if 'intent' in context:
            intent = context['intent']
            response_type = intent.get('response_type', 'conversational')
            system_parts.append(f"This appears to be a {response_type} request.")
        
        # Add topic context
        if 'topics' in context:
            topics = context['topics']
            if topics.get('topics'):
                top_topics = [t['word'] for t in topics['topics'][:3]]
                system_parts.append(f"Key topics: {', '.join(top_topics)}")
        
        system_prompt = " ".join(system_parts)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add memory context
        if 'memories' in context:
            for mem in context['memories'][:5]:
                messages.append({
                    "role": "assistant",
                    "content": f"[Previous context] {mem.get('content', '')[:300]}"
                })
        
        # Add web context
        if 'web_results' in context:
            for result in context['web_results'][:3]:
                messages.append({
                    "role": "assistant",
                    "content": f"[Web: {result.get('title', '')}] {result.get('body', '')[:200]}"
                })
        
        # Add tool results
        if 'tool_results' in context:
            for tr in context['tool_results']:
                messages.append({
                    "role": "assistant",
                    "content": f"[Tool: {tr.get('tool', '')}] {json.dumps(tr.get('output', ''))[:300]}"
                })
        
        messages.append({"role": "user", "content": query})
        
        return self.llm.chat(messages, temperature=temperature, max_tokens=max_tokens)

# =============================================================================
# Plugin System
# =============================================================================

class PluginManager:
    """Manages plugins and extensions."""
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.hooks: Dict[str, List[Callable]] = {
            'pre_process': [],
            'post_process': [],
            'pre_response': [],
            'post_response': []
        }
    
    def register_plugin(self, name: str, plugin: Any):
        """Register a plugin."""
        self.plugins[name] = plugin
        logger.info(f"Registered plugin: {name}")
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback."""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(callback)
    
    async def execute_hooks(self, hook_name: str, data: Any) -> Any:
        """Execute all callbacks for a hook."""
        result = data
        for callback in self.hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(result)
                else:
                    result = callback(result)
            except Exception as e:
                logger.error(f"Hook {hook_name} callback failed: {e}")
        return result

# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window_seconds
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests."""
        if client_id not in self.requests:
            return self.max_requests
        
        now = time.time()
        recent = [t for t in self.requests[client_id] if now - t < self.window_seconds]
        return max(0, self.max_requests - len(recent))

# =============================================================================
# Health Monitor
# =============================================================================

class HealthMonitor:
    """Monitors system health."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
    
    def record_request(self):
        """Record a request."""
        self.request_count += 1
    
    def record_error(self, error: str):
        """Record an error."""
        self.error_count += 1
        self.last_error = error
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        uptime = time.time() - self.start_time
        
        return {
            'status': 'healthy' if self.error_count < self.request_count * 0.1 else 'degraded',
            'uptime_seconds': uptime,
            'uptime_formatted': self._format_uptime(uptime),
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'last_error': self.last_error
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human readable."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        
        return " ".join(parts) if parts else "<1m"

# =============================================================================
# Additional API Endpoints
# =============================================================================

# Initialize additional components
sentiment_analyzer = SentimentAnalyzer()
intent_classifier = IntentClassifier()
topic_extractor = TopicExtractor()
health_monitor = HealthMonitor()
rate_limiter = RateLimiter()
plugin_manager = PluginManager()
conversation_manager: Optional[ConversationManager] = None

@app.on_event("startup")
async def startup_additional():
    global conversation_manager
    if engine:
        conversation_manager = ConversationManager(engine.db)

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return health_monitor.get_health()

@app.post("/analyze/sentiment")
async def analyze_sentiment(text: str):
    """Analyze text sentiment."""
    return sentiment_analyzer.analyze(text)

@app.post("/analyze/intent")
async def analyze_intent(text: str):
    """Classify text intent."""
    return intent_classifier.classify(text)

@app.post("/analyze/topics")
async def analyze_topics(text: str):
    """Extract topics from text."""
    return topic_extractor.extract(text)

@app.post("/analyze/full")
async def full_analysis(text: str):
    """Perform full text analysis."""
    return {
        'sentiment': sentiment_analyzer.analyze(text),
        'intent': intent_classifier.classify(text),
        'topics': topic_extractor.extract(text)
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    health_monitor.record_request()
    
    async def generate():
        try:
            # Process through loops first
            loop_context = {}
            for loop_type, processor in engine.loops.items():
                try:
                    loop_context[loop_type.value] = processor.process(request.message, loop_context)
                except Exception:
                    pass
            
            # Build messages
            messages = [
                {"role": "system", "content": "You are gpc3, an advanced AI assistant."},
                {"role": "user", "content": request.message}
            ]
            
            # Stream response
            streamer = StreamingHandler(request.model or engine.config.get("default_model", "qwen2:7b"))
            
            async for chunk in streamer.stream_chat(
                messages,
                temperature=loop_context.get('self_adjustment', {}).get('temperature', 0.7)
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        
        except Exception as e:
            health_monitor.record_error(str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.get("/conversations")
async def list_conversations():
    """List active conversations."""
    if not conversation_manager:
        return {"conversations": []}
    
    return {
        "conversations": list(conversation_manager.conversations.keys())
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not initialized")
    
    history = conversation_manager.get_history(conversation_id)
    return {"conversation_id": conversation_id, "history": history}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_manager:
        conversation_manager.clear_conversation(conversation_id)
    return {"status": "deleted"}

@app.get("/plugins")
async def list_plugins():
    """List registered plugins."""
    return {
        "plugins": list(plugin_manager.plugins.keys()),
        "hooks": {k: len(v) for k, v in plugin_manager.hooks.items()}
    }

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPC3 AI Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "gpc3_core:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
