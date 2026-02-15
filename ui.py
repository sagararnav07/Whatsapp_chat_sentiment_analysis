"""
ui.py — Streamlit UI Components for ChatSense AI
=================================================
Reusable UI components:
  - Chat panel with streaming responses
  - Evidence display panel
  - Stats cards
  - Query type badges
"""

import streamlit as st
from typing import List, Dict, Optional


def render_evidence_panel(evidence: List[dict]):
    """Render a collapsible evidence panel showing supporting messages."""
    if not evidence:
        return

    with st.expander(f"📎 Supporting Evidence ({len(evidence)} items)", expanded=False):
        for i, ev in enumerate(evidence):
            ev_type = ev.get('type', 'unknown')

            # Icon by type
            icon = {
                'filtered': '🔍',
                'semantic': '🧠',
                'desire': '💭',
            }.get(ev_type, '📌')

            score_str = f"  •  relevance: {ev['score']:.2f}" if 'score' in ev else ""

            st.markdown(
                f"{icon} **[{ev.get('date_str', '?')}]** **{ev.get('speaker', '?')}**: "
                f"_{ev.get('message', '')[:200]}_"
                f"{score_str}"
            )
            if i < len(evidence) - 1:
                st.markdown("---")


def render_query_badge(query_analysis: dict):
    """Show a small badge indicating how the query was interpreted."""
    qt = query_analysis.get('query_type', 'general')
    badges = {
        'date_speaker': '📅👤 Date + Speaker filtered',
        'date': '📅 Date filtered',
        'speaker': '👤 Speaker filtered',
        'desire': '💭 Desire/Intent detection',
        'summary': '📝 Conversation summary',
        'emotional_spike': '⚡ Emotional spike detection',
        'semantic': '🧠 Semantic search',
        'general': '💬 General query',
    }
    label = badges.get(qt, qt)

    parts = [label]
    if query_analysis.get('speaker'):
        parts.append(f"Speaker: **{query_analysis['speaker']}**")
    if query_analysis.get('start_date'):
        date_range = query_analysis['start_date']
        if query_analysis.get('end_date') and query_analysis['end_date'] != query_analysis['start_date']:
            date_range += f" → {query_analysis['end_date']}"
        parts.append(f"Date: **{date_range}**")

    st.caption(" • ".join(parts))


def render_welcome_message() -> dict:
    """Return the welcome message for the chat."""
    return {
        "role": "assistant",
        "content": (
            "👋 Hey! I'm **ChatSense AI** — your conversational intelligence engine.\n\n"
            "**What I can do:**\n"
            "- 📅 **Date queries**: \"What did Arnav say on 15th March?\"\n"
            "- 👤 **Speaker queries**: \"What does Kartik talk about most?\"\n"
            "- 🧠 **Semantic search**: \"Find messages about food\" \n"
            "- 💭 **Desire detection**: \"Did Arnav ever want something?\"\n"
            "- ⚡ **Emotional analysis**: \"When did arguments happen?\"\n"
            "- 📝 **Summaries**: \"Summarize conversation in October 2025\"\n"
            "- 🔍 **Deep inference**: \"What does the chat reveal about their relationship?\"\n\n"
            "Every response includes **evidence** — real quotes with dates.\n\n"
            "Upload a chat to get started! 🚀"
        ),
        "evidence": [],
        "query_analysis": {},
    }


def render_stats_cards(stats: dict):
    """Render quick stat metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 Messages", f"{stats.get('total_messages', 0):,}")
    with col2:
        st.metric("📝 Words", f"{stats.get('total_words', 0):,}")
    with col3:
        st.metric("🖼️ Media", f"{stats.get('media_count', 0):,}")
    with col4:
        st.metric("📅 Duration", f"{stats.get('duration_days', 0)} days")


def render_index_status(engine):
    """Show a status bar about the index."""
    if engine is None:
        st.info("📁 Upload a WhatsApp chat export to begin analysis")
        return

    chunks = engine.chunk_index.total_chunks
    msgs = len(engine.text_df)
    speakers = ', '.join(engine.speakers)
    st.success(
        f"✅ **Indexed**: {msgs:,} messages in {chunks} chunks • "
        f"Speakers: {speakers}"
    )
