"""
embeddings.py — FAISS-backed Semantic Embedding Engine
======================================================
Responsibilities:
  1. Chunk parsed messages into contextual windows (time-based or count-based).
  2. Compute dense embeddings via sentence-transformers.
  3. Build and query a FAISS index for fast semantic retrieval.
  4. Attach rich metadata (speaker, timestamp range, message count) to each chunk.

Architecture notes:
  - Uses `all-MiniLM-L6-v2` (384-dim, fast, good quality).
  - Chunks are formed by *conversation sessions*: a new session starts when
    there is a gap > 30 min between messages, OR every 150 messages.
  - Each chunk stores the full concatenated text plus metadata for filtering.
"""

import numpy as np
import pandas as pd
import faiss
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import timedelta

# Lazy-loaded model (singleton)
_model = None


def _get_model():
    """Load the sentence-transformer model (cached as a module-level singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


# ── Chunk dataclass ──────────────────────────────────────────────────────────

@dataclass
class ChatChunk:
    """A chunk of conversation messages with metadata."""
    chunk_id: int
    text: str                   # Concatenated "Speaker: message\n" lines
    messages: List[dict]        # Raw row dicts from the DataFrame
    speakers: List[str]         # Unique speakers in this chunk
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    message_count: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def time_range_str(self) -> str:
        return f"{self.start_time.strftime('%d %b %Y %H:%M')} – {self.end_time.strftime('%d %b %Y %H:%M')}"


# ── Chunking logic ──────────────────────────────────────────────────────────

def create_chunks(
    df: pd.DataFrame,
    max_messages: int = 150,
    session_gap_minutes: int = 30,
) -> List[ChatChunk]:
    """
    Split the parsed DataFrame into contextual chunks.

    Strategy:
      - Walk through messages in order. Start a new chunk when:
        a) The time gap from the previous message > session_gap_minutes, OR
        b) The current chunk reaches max_messages.
      - Each chunk contains the full text plus metadata.
    """
    if df.empty:
        return []

    chunks: List[ChatChunk] = []
    current_msgs: List[dict] = []
    chunk_id = 0
    prev_time = None

    for _, row in df.iterrows():
        ts = row['timestamp']

        # Decide if we should start a new chunk
        start_new = False
        if prev_time is not None:
            gap = (ts - prev_time).total_seconds() / 60
            if gap > session_gap_minutes:
                start_new = True
        if len(current_msgs) >= max_messages:
            start_new = True

        # Flush current chunk
        if start_new and current_msgs:
            chunks.append(_build_chunk(chunk_id, current_msgs))
            chunk_id += 1
            current_msgs = []

        current_msgs.append(row.to_dict())
        prev_time = ts

    # Flush last chunk
    if current_msgs:
        chunks.append(_build_chunk(chunk_id, current_msgs))

    return chunks


def _build_chunk(chunk_id: int, messages: List[dict]) -> ChatChunk:
    """Create a ChatChunk from a list of message dicts."""
    lines = []
    for m in messages:
        ts_str = m['timestamp'].strftime('%d/%m/%Y %H:%M')
        lines.append(f"[{ts_str}] {m['speaker']}: {m['message']}")

    text = "\n".join(lines)
    speakers = list({m['speaker'] for m in messages if m['speaker'] != 'group_notification'})
    timestamps = [m['timestamp'] for m in messages]

    return ChatChunk(
        chunk_id=chunk_id,
        text=text,
        messages=messages,
        speakers=speakers,
        start_time=min(timestamps),
        end_time=max(timestamps),
        message_count=len(messages),
    )


# ── Embedding computation ───────────────────────────────────────────────────

def compute_chunk_embeddings(chunks: List[ChatChunk], batch_size: int = 64) -> np.ndarray:
    """Compute embeddings for all chunks; also sets each chunk.embedding."""
    model = _get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=batch_size)
    embeddings = np.array(embeddings, dtype='float32')

    # Normalize for cosine similarity via inner-product
    faiss.normalize_L2(embeddings)

    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]

    return embeddings


def compute_query_embedding(query: str) -> np.ndarray:
    """Compute a normalized embedding for a query string."""
    model = _get_model()
    emb = model.encode([query], show_progress_bar=False)
    emb = np.array(emb, dtype='float32')
    faiss.normalize_L2(emb)
    return emb


# ── FAISS index ──────────────────────────────────────────────────────────────

class ChatIndex:
    """FAISS index wrapping chunks with metadata-aware retrieval."""

    def __init__(self, chunks: List[ChatChunk]):
        self.chunks = chunks
        self._index: Optional[faiss.IndexFlatIP] = None
        self._build()

    def _build(self):
        """Build the FAISS index from chunk embeddings."""
        if not self.chunks:
            return

        embeddings = compute_chunk_embeddings(self.chunks)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner product (== cosine after normalization)
        self._index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_score: float = 0.10,
    ) -> List[Tuple[ChatChunk, float]]:
        """
        Semantic search with optional metadata pre-filtering.

        Returns list of (chunk, score) tuples sorted by relevance.
        """
        if self._index is None or not self.chunks:
            return []

        query_emb = compute_query_embedding(query)

        # Search more than top_k so we can filter afterward
        search_k = min(len(self.chunks), top_k * 5)
        scores, indices = self._index.search(query_emb, search_k)

        results: List[Tuple[ChatChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < min_score:
                continue

            chunk = self.chunks[idx]

            # Speaker filter
            if speaker and speaker not in chunk.speakers:
                continue

            # Date range filter
            if start_date:
                sd = pd.to_datetime(start_date)
                if chunk.end_time < sd:
                    continue
            if end_date:
                ed = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                if chunk.start_time > ed:
                    continue

            results.append((chunk, float(score)))

            if len(results) >= top_k:
                break

        return results

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_messages(self) -> int:
        return sum(c.message_count for c in self.chunks)


# ── Per-message index for fine-grained retrieval ─────────────────────────────

class MessageIndex:
    """
    A per-message FAISS index for fine-grained retrieval.
    Stores individual messages with their metadata.
    Used when we need to find specific statements.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Build index from parsed DataFrame.
        Only indexes text messages (not media/system).
        """
        self.df = df.copy()
        self._text_df = df[(~df.get('is_media', False)) & (df['speaker'] != 'group_notification')].copy()
        self._text_df = self._text_df[self._text_df['message'].str.strip().astype(bool)].reset_index(drop=True)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None
        self._build()

    def _build(self):
        if self._text_df.empty:
            return
        model = _get_model()
        # Build searchable text: "Speaker: message"
        texts = (self._text_df['speaker'] + ': ' + self._text_df['message']).tolist()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=128)
        embeddings = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)
        self._embeddings = embeddings

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 15,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_score: float = 0.15,
    ) -> List[dict]:
        """
        Search individual messages.  Returns list of dicts with
        speaker, message, timestamp, score.
        """
        if self._index is None or self._text_df.empty:
            return []

        query_emb = compute_query_embedding(query)
        search_k = min(len(self._text_df), top_k * 5)
        scores, indices = self._index.search(query_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue

            row = self._text_df.iloc[idx]

            # Metadata filters
            if speaker and row['speaker'] != speaker:
                continue
            if start_date:
                sd = pd.to_datetime(start_date)
                if row['timestamp'] < sd:
                    continue
            if end_date:
                ed = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                if row['timestamp'] > ed:
                    continue

            results.append({
                'speaker': row['speaker'],
                'message': row['message'],
                'timestamp': row['timestamp'],
                'date_str': row['timestamp'].strftime('%d %b %Y, %H:%M'),
                'score': float(score),
            })

            if len(results) >= top_k:
                break

        return results
