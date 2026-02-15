"""
reasoning.py — LLM Reasoning Layer (Groq / Llama 3.3)
======================================================
Responsibilities:
  1. Build precision system prompts based on query type.
  2. Assemble retrieved context into the optimal prompt structure.
  3. Call Groq API (Llama 3.3 70B) with streaming support.
  4. Enforce evidence-based responses with anti-hallucination rules.

The LLM never sees the entire chat — only the relevant, retrieved context.
This enables handling 10 MB+ files efficiently.
"""

import os
import json
from typing import Optional, List, Dict, Any, Generator
from groq import Groq

from retrieval import RetrievalEngine


# ── API setup ────────────────────────────────────────────────────────────────

def _get_groq_client() -> Optional[Groq]:
    """Get Groq client from env or Streamlit secrets."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if api_key:
        return Groq(api_key=api_key)
    return None


# ── System prompt templates ──────────────────────────────────────────────────

_BASE_SYSTEM = """You are **ChatSense AI** — an expert conversational intelligence analyst.
You have access to a WhatsApp chat that has been parsed, indexed, and semantically searched.
Below is the retrieved context that is relevant to the user's question.

{stats}

{context}

## YOUR RULES (MANDATORY):
1. **Always cite evidence.** Quote the actual message with [date] Speaker: "message".
2. **Always mention the date** when referencing a specific message.
3. **Always mention the speaker** name.
4. **Explain your reasoning** — how you arrived at your conclusion.
5. **If no messages from the queried speaker on the queried date**, explain this clearly:
   - State that the speaker did not send any text messages on that date.
   - If other speakers sent messages on that date, mention what they sent.
   - If media was shared (images, documents), mention that too.
   - If there are messages from nearby dates, mention those.
6. **Never hallucinate** — do not invent messages, dates, or speakers that aren't in the context.
7. **Be specific** — use exact numbers, dates, percentages from the data.
8. **Always provide useful information** — even when the exact query yields no results,
   tell the user what IS available (nearby dates, other speakers, media activity).

## RESPONSE FORMAT:
- Start with a direct answer to the question.
- If the answer is "no messages found", explain WHY and what WAS found.
- Provide supporting evidence with quoted messages.
- Add brief analysis/inference where appropriate.
- Keep responses focused and evidence-based.
"""

_DESIRE_SYSTEM = """You are **ChatSense AI** — an expert in detecting implicit desires, emotional needs, and unspoken intentions in conversations.

{stats}

{context}

## YOUR SPECIALIZED TASK:
You are analyzing messages for **implicit desires, wishes, and emotional needs**.

Look for:
- Direct expressions: "I wish...", "I want...", "I need..."
- Indirect expressions: "It would be nice if...", "If only..."
- Passive desire: Complaints that imply a wish for change
- Emotional undertones: Loneliness, frustration, longing, craving
- Behavioral signals: Repeated topics, escalation, withdrawal

## RESPONSE FORMAT:
For each finding, provide:
1. **Date & Speaker**: When and who said it
2. **Quote**: The exact message
3. **Interpretation**: What desire/need this reveals
4. **Confidence**: High/Medium/Low based on directness

Example:
> On **13 May 2025 at 22:14**, **Arnav** said: *"I just feel like no one understands me lately."*
> **Interpretation**: This suggests a passive emotional need for reassurance, empathy, or deeper connection. The phrase "no one understands" indicates feeling isolated.
> **Confidence**: Medium (indirect expression)

## RULES:
1. Only reference messages that exist in the context above.
2. If no desire/intent expressions are found, say so clearly.
3. Distinguish between direct and indirect/implicit expressions.
4. Never invent messages.
"""

_SUMMARY_SYSTEM = """You are **ChatSense AI** — an expert conversation summarizer.

{stats}

{context}

## YOUR TASK:
Provide a comprehensive summary of the conversation in the given context.

Include:
1. **Main topics** discussed
2. **Key events** or decisions
3. **Emotional tone** of the conversation
4. **Notable exchanges** (quote specific messages)
5. **Relationship dynamics** observed

## RULES:
1. Only reference messages in the context.
2. Quote specific messages as evidence.
3. Mention dates and speakers.
4. If the date range contains no messages, say so.
"""

_EMOTIONAL_SYSTEM = """You are **ChatSense AI** — an expert in detecting emotional spikes, conflicts, and tension in conversations.

{stats}

{context}

## YOUR TASK:
Identify emotional high points, arguments, and tension in the conversation.

Look for:
- Insults, sarcasm, or aggressive language
- Rapid-fire exchanges (many messages in short time)
- ALL CAPS or excessive punctuation
- Defensive or accusatory tone
- Abrupt topic changes or silence after conflict
- Resolution or reconciliation patterns

## RESPONSE FORMAT:
For each emotional spike:
1. **When**: Date and time range
2. **Who**: Participants involved
3. **What happened**: Brief description with quoted messages
4. **Intensity**: Low/Medium/High
5. **Resolution**: Was it resolved? How?

## RULES:
1. Only reference messages that exist in the context.
2. Quote exact messages as evidence.
3. Be balanced — don't take sides.
4. If no emotional spikes found, say so clearly.
"""


def _select_system_prompt(query_type: str) -> str:
    """Select the best system prompt template based on query type."""
    if query_type == 'desire':
        return _DESIRE_SYSTEM
    elif query_type == 'summary':
        return _SUMMARY_SYSTEM
    elif query_type == 'emotional_spike':
        return _EMOTIONAL_SYSTEM
    else:
        return _BASE_SYSTEM


# ── Main reasoning function ─────────────────────────────────────────────────

def reason(
    query: str,
    engine: RetrievalEngine,
    conversation_history: Optional[List[Dict]] = None,
    stream: bool = False,
) -> Any:
    """
    Full reasoning pipeline:
      1. Retrieve relevant context via the retrieval engine.
      2. Build an optimized prompt.
      3. Call the LLM.
      4. Return the response (or a generator if streaming).

    Returns
    -------
    If stream=False: dict with 'response', 'evidence', 'query_analysis'
    If stream=True: dict with 'stream' (generator), 'evidence', 'query_analysis'
    """
    client = _get_groq_client()
    if client is None:
        return {
            'response': "❌ AI not configured. Set GROQ_API_KEY in your environment or Streamlit secrets.",
            'evidence': [],
            'query_analysis': {},
        }

    # Step 1: Retrieve context
    retrieval_result = engine.retrieve(query, top_k=15)
    qa = retrieval_result['query_analysis']
    context = retrieval_result['context_text']
    evidence = retrieval_result['evidence']

    # Step 2: Build prompt
    stats = engine.get_conversation_stats()
    template = _select_system_prompt(qa['query_type'])
    system_prompt = template.format(stats=stats, context=context)

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 8 exchanges for memory)
    if conversation_history:
        recent = conversation_history[-8:]
        for msg in recent:
            if msg.get('role') in ('user', 'assistant'):
                messages.append({
                    "role": msg['role'],
                    "content": msg['content'][:600],
                })

    messages.append({"role": "user", "content": query})

    # Step 3: Call LLM
    if stream:
        gen = _stream_response(client, messages)
        return {
            'stream': gen,
            'evidence': evidence,
            'query_analysis': qa,
        }
    else:
        response_text = _call_llm(client, messages)
        return {
            'response': response_text,
            'evidence': evidence,
            'query_analysis': qa,
        }


def _call_llm(client: Groq, messages: list) -> str:
    """Non-streaming LLM call."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,        # Lower temp for factual accuracy
            max_tokens=2000,
            top_p=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {str(e)}"


def _stream_response(client: Groq, messages: list) -> Generator[str, None, None]:
    """Streaming LLM call — yields text chunks."""
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=2000,
            top_p=0.9,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n\n❌ AI Error: {str(e)}"


# ── Quick analysis helpers (no LLM needed) ──────────────────────────────────

def get_quick_stats(engine: RetrievalEngine) -> str:
    """Return formatted statistics without calling the LLM."""
    return engine.get_conversation_stats()
