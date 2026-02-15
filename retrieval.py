"""
retrieval.py — Multi-layer Retrieval Engine
============================================
Implements a 3-stage retrieval pipeline:

  Stage 1 — Structured Filtering
    Filter by date range, speaker, or both using Pandas.

  Stage 2 — Semantic Retrieval
    Use FAISS (via embeddings.py) to find the most relevant chunks/messages.

  Stage 3 — Context Assembly
    Merge results from both stages, deduplicate, rank, and package
    the context for the LLM reasoning layer.

This module coordinates between parser.py and embeddings.py to deliver
the right context to reasoning.py.
"""

import re
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

from parser import (
    parse_chat, get_text_messages, get_speakers,
    filter_by_speaker, filter_by_date_range,
)
from embeddings import (
    create_chunks, ChatIndex, MessageIndex, ChatChunk,
)


# ── Intent patterns for user queries ────────────────────────────────────────

# Date extraction patterns
_DATE_PATTERNS = [
    # "15th March 2025", "15 March 2025"
    r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
    # "March 15, 2025"
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})',
    # "15/03/2025", "15-03-2025"
    r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})',
    # "March 2025" (month only)
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
]

# Date range patterns
_RANGE_PATTERNS = [
    r'between\s+(.+?)\s+and\s+(.+?)(?:\s*[,.]|$)',
    r'from\s+(.+?)\s+to\s+(.+?)(?:\s*[,.]|$)',
    r'during\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
]

# Speaker extraction pattern
_SPEAKER_RE = re.compile(
    r'(?:what did|did|when did|has|show me|find)\s+(\w[\w\s]*?)(?:\s+(?:say|said|mention|express|talk|write|send|ask|want|wish|like|ever|between|on|in|during|from))',
    re.IGNORECASE,
)

# Intent/desire patterns — things people might "want" or "wish for"
DESIRE_PATTERNS = [
    r'\b(?:i\s+)?wish\b',
    r'\b(?:i\s+)?want(?:ed|s)?\b',
    r'\bwould\s+(?:be\s+)?(?:nice|love|like|great)\b',
    r'\bif\s+only\b',
    r'\bplease\b',
    r'\bgive\s+me\b',
    r'\bneed(?:s|ed)?\b',
    r'\bhope(?:s|d)?\b',
    r'\bcraving\b',
    r'\bmissing\b',
    r'\bfeel\s+like\b',
    r'\bi\s+feel\b',
    r'\bjust\s+feel\b',
    r'\bno\s+one\b',
    r'\blonely\b',
    r'\balone\b',
]


# ── Query analysis ──────────────────────────────────────────────────────────

def analyze_query(query: str, speakers: List[str]) -> Dict[str, Any]:
    """
    Analyze a natural-language query to extract structured filters.

    Returns dict with:
      - speaker:     str or None
      - start_date:  str (ISO) or None
      - end_date:    str (ISO) or None
      - is_desire_query: bool
      - is_date_query: bool
      - is_speaker_query: bool
      - query_type:  str  ('date', 'speaker', 'semantic', 'desire', 'summary', 'general')
      - cleaned_query: str (for semantic search)
    """
    result = {
        'speaker': None,
        'start_date': None,
        'end_date': None,
        'is_desire_query': False,
        'is_date_query': False,
        'is_speaker_query': False,
        'query_type': 'general',
        'cleaned_query': query,
    }

    q_lower = query.lower()

    # ── Extract speaker ──
    # First try exact match against known speakers
    for sp in speakers:
        if sp.lower() in q_lower:
            result['speaker'] = sp
            result['is_speaker_query'] = True
            break

    # If no exact match, try pattern
    if not result['speaker']:
        m = _SPEAKER_RE.search(query)
        if m:
            candidate = m.group(1).strip()
            # Fuzzy match against known speakers
            for sp in speakers:
                if candidate.lower() in sp.lower() or sp.lower() in candidate.lower():
                    result['speaker'] = sp
                    result['is_speaker_query'] = True
                    break

    # ── Extract date range ──
    for pat in _RANGE_PATTERNS:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                start_str, end_str = groups
                sd = _parse_flexible_date(start_str)
                ed = _parse_flexible_date(end_str)
                if sd:
                    result['start_date'] = sd
                if ed:
                    result['end_date'] = ed
                result['is_date_query'] = True
                break

    # ── Extract single date ──
    if not result['is_date_query']:
        for pat in _DATE_PATTERNS:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                date_str = m.group(0)
                parsed = _parse_flexible_date(date_str)
                if parsed:
                    result['start_date'] = parsed
                    result['end_date'] = parsed
                    result['is_date_query'] = True
                break

    # ── Detect desire/intent queries ──
    desire_keywords = ['want', 'wish', 'desire', 'need', 'hope', 'craving',
                       'feeling', 'emotional', 'express', 'implicit', 'indirect',
                       'hug', 'comfort', 'love', 'lonely', 'miss']
    if any(kw in q_lower for kw in desire_keywords):
        result['is_desire_query'] = True

    # ── Determine query type ──
    if 'summarize' in q_lower or 'summary' in q_lower or 'overview' in q_lower:
        result['query_type'] = 'summary'
    elif 'argument' in q_lower or 'fight' in q_lower or 'conflict' in q_lower or 'angry' in q_lower:
        result['query_type'] = 'emotional_spike'
    elif result['is_desire_query']:
        result['query_type'] = 'desire'
    elif result['is_date_query'] and result['is_speaker_query']:
        result['query_type'] = 'date_speaker'
    elif result['is_date_query']:
        result['query_type'] = 'date'
    elif result['is_speaker_query']:
        result['query_type'] = 'speaker'
    else:
        result['query_type'] = 'semantic'

    return result


def _parse_flexible_date(text: str) -> Optional[str]:
    """Try to parse a date string into ISO format (YYYY-MM-DD)."""
    text = text.strip().rstrip(',.')
    # Remove ordinal suffixes
    text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)

    formats = [
        '%d %B %Y', '%B %d %Y', '%B %d, %Y',
        '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y',
        '%d/%m/%y', '%d-%m-%y', '%m/%d/%y',
        '%B %Y',   # Month + year only
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == '%B %Y':
                # Return first day of month
                return dt.strftime('%Y-%m-01')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # Fallback to pandas
    try:
        dt = pd.to_datetime(text, dayfirst=True)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None


# ── Multi-layer retrieval engine ─────────────────────────────────────────────

class RetrievalEngine:
    """
    Orchestrates multi-layer retrieval:
      1. Structured filter (date/speaker via Pandas)
      2. Semantic search (FAISS)
      3. Context assembly for LLM
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with parsed chat DataFrame.
        Builds chunk index + message index.
        """
        self.df = df
        self.text_df = get_text_messages(df)
        self.speakers = get_speakers(df)

        # Build indices
        self.chunks = create_chunks(df)
        self.chunk_index = ChatIndex(self.chunks)
        self.message_index = MessageIndex(df)

    def retrieve(self, query: str, top_k: int = 15) -> Dict[str, Any]:
        """
        Full retrieval pipeline with intelligent fallback.

        Returns dict with:
          - query_analysis: parsed query metadata
          - filtered_messages: messages matching structured filters
          - semantic_results: semantically similar messages
          - context_text: assembled context string for LLM
          - evidence: list of evidence dicts for UI display
        """
        # Stage 1: Analyze the query
        qa = analyze_query(query, self.speakers)

        # Stage 2: Structured filtering with FALLBACK strategy
        filtered_df = self.text_df.copy()
        date_all_df = pd.DataFrame()       # All messages on date (any speaker)
        date_full_df = pd.DataFrame()      # All messages inc. media on date
        fallback_used = None               # Track which fallback was used

        if qa['speaker']:
            filtered_df = filtered_df[filtered_df['speaker'] == qa['speaker']]
        if qa['start_date']:
            filtered_df = filter_by_date_range(filtered_df, qa['start_date'], qa.get('end_date'))
        elif qa['end_date']:
            filtered_df = filter_by_date_range(filtered_df, end_date=qa['end_date'])

        # FALLBACK 1: If speaker+date returns empty, try date-only (all speakers)
        if filtered_df.empty and qa['is_date_query'] and qa['speaker']:
            date_all_df = filter_by_date_range(self.text_df, qa['start_date'], qa.get('end_date'))
            if not date_all_df.empty:
                fallback_used = 'date_only_text'
                filtered_df = date_all_df

        # FALLBACK 2: If still empty, check ALL messages including media on that date
        if filtered_df.empty and qa['is_date_query']:
            date_full_df = filter_by_date_range(self.df, qa['start_date'], qa.get('end_date'))
            if not date_full_df.empty:
                fallback_used = 'date_all_messages'

        # FALLBACK 3: If date query with speaker but no date results,
        # try showing the speaker's messages around that time (+-3 days)
        if filtered_df.empty and date_full_df.empty and qa['is_date_query'] and qa['start_date']:
            try:
                target_date = pd.to_datetime(qa['start_date'])
                nearby_start = (target_date - timedelta(days=3)).strftime('%Y-%m-%d')
                nearby_end = (target_date + timedelta(days=3)).strftime('%Y-%m-%d')
                nearby_df = filter_by_date_range(self.text_df, nearby_start, nearby_end)
                if qa['speaker']:
                    speaker_nearby = nearby_df[nearby_df['speaker'] == qa['speaker']]
                    if not speaker_nearby.empty:
                        filtered_df = speaker_nearby
                        fallback_used = 'nearby_dates'
                    elif not nearby_df.empty:
                        filtered_df = nearby_df
                        fallback_used = 'nearby_dates_all'
                elif not nearby_df.empty:
                    filtered_df = nearby_df
                    fallback_used = 'nearby_dates'
            except Exception:
                pass

        qa['_fallback_used'] = fallback_used
        qa['_date_full_df'] = date_full_df  # All messages inc. media for context

        # Stage 3: Semantic search (on both chunk and message level)
        # First try with all filters
        chunk_results = self.chunk_index.search(
            query, top_k=5,
            speaker=qa['speaker'],
            start_date=qa['start_date'],
            end_date=qa.get('end_date'),
        )
        message_results = self.message_index.search(
            query, top_k=top_k,
            speaker=qa['speaker'],
            start_date=qa['start_date'],
            end_date=qa.get('end_date'),
        )

        # SEMANTIC FALLBACK: If semantic returned nothing with filters, broaden search
        if not message_results and (qa['speaker'] or qa['is_date_query']):
            # Try without speaker filter first
            message_results = self.message_index.search(
                query, top_k=top_k,
                speaker=None,
                start_date=qa['start_date'],
                end_date=qa.get('end_date'),
            )
        if not message_results:
            # Try completely unfiltered
            message_results = self.message_index.search(
                query, top_k=top_k,
            )
        if not chunk_results and (qa['speaker'] or qa['is_date_query']):
            chunk_results = self.chunk_index.search(
                query, top_k=5,
                speaker=None,
                start_date=qa['start_date'],
                end_date=qa.get('end_date'),
            )

        # Stage 3b: If desire query, also search for desire patterns
        desire_matches = []
        if qa['is_desire_query']:
            desire_matches = self._find_desire_expressions(
                filtered_df if len(filtered_df) < len(self.text_df) else self.text_df,
                qa['speaker'],
            )

        # Stage 4: Assemble context
        context_text, evidence = self._assemble_context(
            qa, filtered_df, chunk_results, message_results, desire_matches
        )

        return {
            'query_analysis': qa,
            'filtered_messages': filtered_df,
            'semantic_results': message_results,
            'desire_matches': desire_matches,
            'chunk_results': chunk_results,
            'context_text': context_text,
            'evidence': evidence,
        }

    def _find_desire_expressions(
        self,
        df: pd.DataFrame,
        speaker: Optional[str] = None,
    ) -> List[dict]:
        """Find messages that express desires, wishes, or emotional needs."""
        if df.empty:
            return []

        subset = df if speaker is None else df[df['speaker'] == speaker]
        matches = []

        combined_pattern = '|'.join(DESIRE_PATTERNS)
        mask = subset['message'].str.contains(combined_pattern, case=False, regex=True, na=False)
        desire_msgs = subset[mask]

        for _, row in desire_msgs.iterrows():
            matches.append({
                'speaker': row['speaker'],
                'message': row['message'],
                'timestamp': row['timestamp'],
                'date_str': row['timestamp'].strftime('%d %b %Y, %H:%M'),
                'type': 'desire_expression',
            })

        return matches

    def _assemble_context(
        self,
        qa: Dict,
        filtered_df: pd.DataFrame,
        chunk_results: List[Tuple[ChatChunk, float]],
        message_results: List[dict],
        desire_matches: List[dict],
    ) -> Tuple[str, List[dict]]:
        """
        Build the context string that will be sent to the LLM,
        plus a list of evidence items for the UI.
        Handles fallback scenarios gracefully.
        """
        parts = []
        evidence = []

        # Header with query understanding
        parts.append(f"## QUERY ANALYSIS")
        parts.append(f"- Type: {qa['query_type']}")
        if qa['speaker']:
            parts.append(f"- Speaker filter: {qa['speaker']}")
        if qa['start_date']:
            parts.append(f"- Date range: {qa['start_date']} to {qa.get('end_date', 'present')}")

        # Indicate fallback if used
        fallback = qa.get('_fallback_used')
        if fallback:
            if fallback == 'date_only_text':
                parts.append(f"- NOTE: **{qa['speaker']}** had NO text messages on this date. Showing messages from ALL speakers on this date.")
            elif fallback == 'date_all_messages':
                parts.append(f"- NOTE: No text messages found on this date. Only media/system messages exist.")
            elif fallback in ('nearby_dates', 'nearby_dates_all'):
                parts.append(f"- NOTE: No messages found on the exact date. Showing nearby messages (±3 days).")
        parts.append("")

        # Show ALL activity on the queried date (including media) for context
        date_full_df = qa.get('_date_full_df', pd.DataFrame())
        if qa['is_date_query'] and not date_full_df.empty:
            parts.append(f"## ALL ACTIVITY ON THIS DATE ({len(date_full_df)} messages total)")
            for _, row in date_full_df.iterrows():
                ts = row['timestamp'].strftime('%d/%m/%Y %H:%M')
                msg = row['message'] if row['message'].strip() else '[empty message]'
                media_tag = ' [MEDIA]' if row.get('is_media', False) else ''
                line = f"[{ts}] {row['speaker']}: {msg}{media_tag}"
                parts.append(line)
                evidence.append({
                    'speaker': row['speaker'],
                    'message': msg + media_tag,
                    'date_str': row['timestamp'].strftime('%d %b %Y, %H:%M'),
                    'type': 'date_activity',
                })
            parts.append("")

        # Filtered messages (show a sample if too many)
        if len(filtered_df) > 0 and qa['is_date_query']:
            fallback_label = ""
            if fallback == 'date_only_text':
                fallback_label = " (from all speakers — original speaker had none)"
            elif fallback in ('nearby_dates', 'nearby_dates_all'):
                fallback_label = " (from nearby dates ±3 days)"
            parts.append(f"## MESSAGES MATCHING FILTERS ({len(filtered_df)} total{fallback_label})")
            sample = filtered_df.head(50)
            for _, row in sample.iterrows():
                ts = row['timestamp'].strftime('%d/%m/%Y %H:%M')
                line = f"[{ts}] {row['speaker']}: {row['message']}"
                parts.append(line)
                evidence.append({
                    'speaker': row['speaker'],
                    'message': row['message'],
                    'date_str': row['timestamp'].strftime('%d %b %Y, %H:%M'),
                    'type': 'filtered',
                })
            if len(filtered_df) > 50:
                parts.append(f"  ... and {len(filtered_df) - 50} more messages")
            parts.append("")

        # Semantic search results
        if message_results:
            parts.append(f"## SEMANTICALLY RELEVANT MESSAGES ({len(message_results)} found)")
            for mr in message_results:
                ts = mr['timestamp'].strftime('%d/%m/%Y %H:%M')
                parts.append(f"[{ts}] {mr['speaker']}: {mr['message']}  (relevance: {mr['score']:.2f})")
                evidence.append({
                    'speaker': mr['speaker'],
                    'message': mr['message'],
                    'date_str': mr['date_str'],
                    'score': mr['score'],
                    'type': 'semantic',
                })
            parts.append("")

        # Chunk context for broader view
        if chunk_results:
            parts.append(f"## CONVERSATION CONTEXT CHUNKS ({len(chunk_results)} chunks)")
            for chunk, score in chunk_results[:3]:   # Top 3 chunks only
                parts.append(f"--- Chunk {chunk.chunk_id} ({chunk.time_range_str}, {chunk.message_count} msgs, score: {score:.2f}) ---")
                # Show first 30 lines of the chunk
                chunk_lines = chunk.text.split('\n')[:30]
                parts.append('\n'.join(chunk_lines))
                if len(chunk.text.split('\n')) > 30:
                    parts.append(f"  ... ({chunk.message_count - 30} more lines)")
                parts.append("")

        # Desire/intent matches
        if desire_matches:
            parts.append(f"## DESIRE/INTENT EXPRESSIONS ({len(desire_matches)} found)")
            for dm in desire_matches:
                ts = dm['timestamp'].strftime('%d/%m/%Y %H:%M')
                parts.append(f"[{ts}] {dm['speaker']}: {dm['message']}")
                evidence.append({
                    'speaker': dm['speaker'],
                    'message': dm['message'],
                    'date_str': dm['date_str'],
                    'type': 'desire',
                })
            parts.append("")

        # If NOTHING was found at all
        has_anything = (
            len(filtered_df) > 0
            or message_results
            or desire_matches
            or chunk_results
            or not date_full_df.empty
        )
        if not has_anything:
            parts.append("## NO MATCHES FOUND")
            if qa['is_date_query'] and qa['speaker']:
                parts.append(f"No messages from {qa['speaker']} found on or near {qa['start_date']}.")
                parts.append("The speaker may not have sent any messages during this period.")
            elif qa['is_date_query']:
                parts.append(f"No messages found on {qa['start_date']}.")
            else:
                parts.append("The query did not match any specific messages.")
            parts.append("Please inform the user of this clearly.")

        context_text = '\n'.join(parts)

        # Limit context size (max ~8000 chars to leave room for system prompt)
        if len(context_text) > 8000:
            context_text = context_text[:8000] + "\n\n... [context truncated for length]"

        # Clean up internal metadata from qa before returning
        qa.pop('_fallback_used', None)
        qa.pop('_date_full_df', None)

        return context_text, evidence

    def get_conversation_stats(self) -> str:
        """Get a stats summary string for the LLM."""
        if self.df.empty:
            return "No data."

        text_df = self.text_df
        total = len(self.df)
        text_count = len(text_df)
        speakers = self.speakers
        first = self.df['timestamp'].min().strftime('%d %b %Y')
        last = self.df['timestamp'].max().strftime('%d %b %Y')
        duration = (self.df['timestamp'].max() - self.df['timestamp'].min()).days

        user_counts = text_df['speaker'].value_counts().to_dict()
        user_lines = '\n'.join(
            f"  - {sp}: {ct} messages ({ct/text_count*100:.1f}%)"
            for sp, ct in user_counts.items()
        )

        total_words = int(text_df['message'].apply(lambda x: len(x.split())).sum())
        media_count = int(self.df['is_media'].sum())

        # Peak hour
        hourly = self.df.groupby(self.df['timestamp'].dt.hour).size()
        peak_hour = int(hourly.idxmax()) if len(hourly) > 0 else 0

        return f"""## CHAT STATISTICS
- Participants: {', '.join(speakers)}
- Total messages: {total:,}
- Text messages: {text_count:,}
- Total words: {total_words:,}
- Media shared: {media_count}
- Date range: {first} to {last} ({duration} days)
- Peak activity hour: {peak_hour}:00

## MESSAGE DISTRIBUTION
{user_lines}

## INDEX INFO
- {self.chunk_index.total_chunks} conversation chunks indexed
- {len(text_df)} messages indexed for semantic search
"""
