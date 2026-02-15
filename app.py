"""
app.py — ChatSense AI: Conversational Intelligence Engine
==========================================================
Main Streamlit application.  Integrates:
  - parser.py     → structured chat parsing
  - embeddings.py → FAISS semantic index
  - retrieval.py  → multi-layer retrieval
  - reasoning.py  → LLM reasoning (Groq / Llama 3.3)
  - ui.py         → reusable UI components
  - helper.py     → legacy analytics (charts, word cloud, etc.)
  - preprocessor.py → legacy preprocessor (for dashboard compatibility)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Legacy modules (kept for dashboard tab compatibility)
import preprocessor
import helper

# New intelligence modules
from parser import parse_chat, get_text_messages, get_speakers, get_stats
from retrieval import RetrievalEngine
from reasoning import reason
from ui import (
    render_evidence_panel,
    render_query_badge,
    render_welcome_message,
    render_index_status,
)

# Rasa Analyser (optional — activated via USE_RASA toggle)
from rasa_service import is_rasa_enabled, is_rasa_healthy, rasa_reason

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatSense AI — Conversational Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    [data-testid="metric-container"] {
        background-color: #21262d; border: 1px solid #30363d;
        border-radius: 10px; padding: 15px;
    }
    [data-testid="stMetricValue"] { color: #58a6ff; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: #161b22;
        border-radius: 10px; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d; border-radius: 8px; color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636; color: white !important;
    }
    .stButton > button {
        background-color: #238636; color: white; border: none;
        border-radius: 6px; padding: 10px 20px; font-weight: 600;
    }
    .stButton > button:hover { background-color: #2ea043; }
    .evidence-box {
        background-color: #161b22; border-left: 3px solid #58a6ff;
        padding: 10px 15px; margin: 5px 0; border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None           # Legacy preprocessor DataFrame
if 'parsed_df' not in st.session_state:
    st.session_state.parsed_df = None    # New parser DataFrame
if 'engine' not in st.session_state:
    st.session_state.engine = None       # RetrievalEngine
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [render_welcome_message()]
if 'show_ai' not in st.session_state:
    st.session_state.show_ai = True
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = None
if 'use_rasa' not in st.session_state:
    st.session_state.use_rasa = is_rasa_enabled()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 ChatSense AI")
    st.caption("Conversational Intelligence Engine")
    st.markdown("---")

    uploaded_file = st.file_uploader("📁 Upload Chat Export", type=['txt'])

    st.markdown("---")
    st.markdown("**How to export (WhatsApp):**")
    st.markdown("1. Open chat → ⋮ → More → Export")
    st.markdown("2. Choose 'Without Media'")
    st.markdown("3. Upload the .txt file")

    st.markdown("---")
    st.markdown("### 🤖 AI Assistant")
    st.session_state.show_ai = st.toggle(
        "Show AI Panel", value=st.session_state.show_ai,
        help="Toggle the AI chat panel",
    )

    # ── Rasa Analyser Toggle ──
    st.markdown("---")
    st.markdown("### 🔬 Rasa Analyser")
    st.session_state.use_rasa = st.toggle(
        "Use Rasa NLP", value=st.session_state.use_rasa,
        help="Route queries through Rasa NLP engine (requires Rasa server)",
    )
    if st.session_state.use_rasa:
        rasa_healthy = is_rasa_healthy()
        if rasa_healthy:
            st.success("🟢 Rasa connected", icon="✅")
        else:
            st.warning("🔴 Rasa server not reachable. Will fallback to ChatSense AI.", icon="⚠️")


# ── Process uploaded file ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Building semantic index...")
def build_engine(raw_text: str) -> RetrievalEngine:
    """Parse the chat and build the retrieval engine (cached)."""
    parsed_df = parse_chat(raw_text)
    return RetrievalEngine(parsed_df)


if uploaded_file is not None:
    raw_text = uploaded_file.getvalue().decode("utf-8")

    # Only rebuild if the file changed
    if st.session_state.raw_text != raw_text:
        st.session_state.raw_text = raw_text

        # New pipeline
        engine = build_engine(raw_text)
        st.session_state.engine = engine
        st.session_state.parsed_df = engine.df

        # Legacy pipeline (for dashboard tabs)
        legacy_df = preprocessor.preprocess(raw_text)
        st.session_state.df = legacy_df
        st.session_state.summary = helper.get_chat_summary(legacy_df)

    engine = st.session_state.engine
    df_legacy = st.session_state.df
    summary = st.session_state.summary
else:
    engine = st.session_state.engine
    df_legacy = st.session_state.df
    summary = st.session_state.summary

# ── Layout ───────────────────────────────────────────────────────────────────
if st.session_state.show_ai:
    main_col, bot_col = st.columns([2, 1])
else:
    main_col = st.container()
    bot_col = None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with main_col:
    if df_legacy is not None and summary is not None:
        user_list = df_legacy['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📊 Chat Analysis Dashboard")
            st.caption(
                f"📅 {summary['date_range']} • {summary['total_days']} days of conversation"
            )
        with col2:
            selected_user = st.selectbox("👤 Select User", user_list)

        st.session_state.selected_user = selected_user
        st.markdown("---")

        # ── Quick Stats ──
        st.subheader("📈 Quick Stats")
        num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df_legacy)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💬 Messages", f"{num_messages:,}")
        c2.metric("📝 Words", f"{words:,}")
        c3.metric("🖼️ Media", f"{num_media:,}")
        c4.metric("🔗 Links", f"{num_links:,}")
        c5.metric("📅 Avg/Day", f"{summary['avg_messages_per_day']}")
        st.markdown("---")

        # ── Tabs ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "💭 Sentiment", "😀 Emojis", "⏰ Activity", "🔍 Deep Dive",
        ])

        # ── TAB 1: OVERVIEW ──
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📅 Monthly Message Trend")
                timeline = helper.monthly_timeline(selected_user, df_legacy)
                fig = px.area(timeline, x='time', y='message', color_discrete_sequence=['#58a6ff'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e', xaxis_title="", yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                )
                fig.update_traces(fill='tozeroy', line=dict(width=2))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### 📆 Daily Activity")
                dtl = helper.daily_timeline(selected_user, df_legacy)
                fig = px.line(dtl, x='only_date', y='message', color_discrete_sequence=['#3fb950'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e', xaxis_title="", yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Activity by Day")
                bd = helper.week_activity_map(selected_user, df_legacy)
                fig = px.bar(x=bd.index, y=bd.values, color=bd.values, color_continuous_scale='Blues')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e', showlegend=False, coloraxis_showscale=False,
                    xaxis_title="", yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### 📊 Activity by Month")
                bm = helper.month_activity_map(selected_user, df_legacy)
                fig = px.bar(x=bm.index, y=bm.values, color=bm.values, color_continuous_scale='Purples')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e', showlegend=False, coloraxis_showscale=False,
                    xaxis_title="", yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig, use_container_width=True)

            if selected_user == 'Overall':
                st.markdown("#### 👥 Most Active Users")
                x, new_df = helper.most_busy_users(df_legacy)
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = px.bar(x=x.index, y=x.values, color=x.values,
                                 color_continuous_scale='Greens',
                                 labels={'x': 'User', 'y': 'Messages'})
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e', showlegend=False, coloraxis_showscale=False,
                        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.dataframe(new_df, use_container_width=True, hide_index=True)

        # ── TAB 2: SENTIMENT ──
        with tab2:
            st.subheader("💭 Sentiment Analysis")
            sentiment_counts, avg_sentiment, sentiment_df = helper.sentiment_analysis(selected_user, df_legacy)
            c1, c2, c3 = st.columns(3)
            total = sentiment_counts.sum()
            with c1:
                pos = sentiment_counts.get('Positive', 0)
                st.metric("😊 Positive", f"{pos:,}",
                          delta=f"{pos/total*100:.1f}%" if total else "0%")
            with c2:
                neu = sentiment_counts.get('Neutral', 0)
                st.metric("😐 Neutral", f"{neu:,}",
                          delta=f"{neu/total*100:.1f}%" if total else "0%")
            with c3:
                neg = sentiment_counts.get('Negative', 0)
                st.metric("😔 Negative", f"{neg:,}",
                          delta=f"{neg/total*100:.1f}%" if total else "0%")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🥧 Sentiment Distribution")
                colors = {'Positive': '#3fb950', 'Neutral': '#d29922', 'Negative': '#f85149'}
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                             color=sentiment_counts.index, color_discrete_map=colors, hole=0.4)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font_color='#8b949e')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if selected_user == 'Overall':
                    st.markdown("#### 👥 User Sentiment Comparison")
                    us = helper.user_sentiment_comparison(df_legacy)
                    clrs = ['#3fb950' if v > 0 else '#f85149' for v in us.values]
                    fig = go.Figure(go.Bar(x=us.values, y=us.index, orientation='h', marker_color=clrs))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e', xaxis_title="Sentiment Score", yaxis_title="",
                        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            mood_label = (
                "🌟 Very Positive" if avg_sentiment > 0.2
                else "😊 Positive" if avg_sentiment > 0.05
                else "😐 Neutral" if avg_sentiment > -0.05
                else "😔 Slightly Negative" if avg_sentiment > -0.2
                else "😢 Negative"
            )
            st.info(f"**Overall Chat Mood:** {mood_label} (Score: {avg_sentiment:.3f})")

        # ── TAB 3: EMOJIS ──
        with tab3:
            st.subheader("😀 Emoji Analysis")
            emoji_df = helper.emoji_analysis(selected_user, df_legacy)
            if not emoji_df.empty:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("#### 🏆 Top Emojis")
                    for _, row in emoji_df.head(10).iterrows():
                        st.markdown(f"**{row['Emoji']}** — {row['Count']} times")
                with c2:
                    st.markdown("#### 📊 Emoji Distribution")
                    fig = px.bar(emoji_df.head(15), x='Emoji', y='Count',
                                 color='Count', color_continuous_scale='Viridis')
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e', coloraxis_showscale=False,
                        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emojis found!")

        # ── TAB 4: ACTIVITY ──
        with tab4:
            st.subheader("⏰ Activity Patterns")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🕐 24-Hour Activity")
                hourly = helper.hourly_activity(selected_user, df_legacy)
                fig = go.Figure(go.Barpolar(
                    r=hourly.values,
                    theta=[f"{h}:00" for h in range(24)],
                    marker_color=hourly.values, marker_colorscale='Blues',
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(showticklabels=False, ticks='', gridcolor='#21262d'),
                        angularaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d'),
                    ),
                    font_color='#8b949e',
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### 🗓️ Weekly Heatmap")
                hm = helper.activity_heatmap(selected_user, df_legacy)
                day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                hm = hm.reindex([d for d in day_order if d in hm.index])
                fig = px.imshow(hm, color_continuous_scale='Blues', aspect='auto')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e',
                )
                st.plotly_chart(fig, use_container_width=True)

            if selected_user == 'Overall':
                st.markdown("#### ⚡ Response Times")
                rt = helper.response_time_analysis(df_legacy)
                if not rt.empty:
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        fig = px.bar(x=rt.index, y=rt.values, color=rt.values,
                                     color_continuous_scale='RdYlGn_r')
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#8b949e', coloraxis_showscale=False,
                            xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        st.success(f"⚡ **Fastest:** {rt.idxmin()}")
                        st.warning(f"🐢 **Slowest:** {rt.idxmax()}")

        # ── TAB 5: DEEP DIVE ──
        with tab5:
            st.subheader("🔍 Deep Dive Analytics")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ☁️ Word Cloud")
                wc = helper.create_wordcloud(selected_user, df_legacy)
                if wc:
                    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0e1117')
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Not enough data")
            with c2:
                st.markdown("#### 📝 Common Words")
                mcdf = helper.most_common_words(selected_user, df_legacy)
                if not mcdf.empty:
                    fig = px.bar(mcdf, x=1, y=0, orientation='h',
                                 color=1, color_continuous_scale='Greens')
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e', coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Chat Streak")
                streak, start, end = helper.get_streak_info(df_legacy)
                st.metric("Consecutive Days", f"🔥 {streak} days")
                st.caption(f"{start} → {end}")
            with c2:
                st.markdown("#### 📅 Most Active Day")
                mad, cnt = helper.get_most_active_date(selected_user, df_legacy)
                st.metric("Date", str(mad))
                st.caption(f"{cnt} messages")

    else:
        # ── Welcome screen ──
        st.title("🧠 ChatSense AI")
        st.markdown("### Conversational Intelligence Engine")
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("### 📅")
            st.markdown("**Date Queries**")
            st.caption("\"What was said on 15th Oct?\"")
        with c2:
            st.markdown("### 💭")
            st.markdown("**Desire Detection**")
            st.caption("\"Did anyone wish for something?\"")
        with c3:
            st.markdown("### 🧠")
            st.markdown("**Semantic Search**")
            st.caption("\"Find messages about money\"")
        with c4:
            st.markdown("### ⚡")
            st.markdown("**Emotional Spikes**")
            st.caption("\"When did arguments happen?\"")
        st.markdown("---")
        st.info("👈 Upload your WhatsApp chat export from the sidebar to get started!")


# ══════════════════════════════════════════════════════════════════════════════
# AI CHAT PANEL — ChatSense AI with Streaming + Evidence
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.show_ai and bot_col is not None:
    with bot_col:
        st.markdown("### 🧠 ChatSense AI")
        st.caption("RAG-powered • Evidence-backed • Streaming responses")

        # Index status
        render_index_status(engine)

        # Chat container
        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.chat_messages[-12:]:
                avatar = "🧠" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    # Show evidence if available
                    ev = msg.get("evidence", [])
                    if ev:
                        render_evidence_panel(ev)
                    # Show query badge
                    qa = msg.get("query_analysis", {})
                    if qa and qa.get("query_type"):
                        render_query_badge(qa)

        # Chat input
        if prompt := st.chat_input("Ask anything about the chat...", key="ai_input"):
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt,
                "evidence": [],
                "query_analysis": {},
            })

            if engine is None:
                # No chat loaded
                response_msg = {
                    "role": "assistant",
                    "content": "📁 Please upload a WhatsApp chat export first! I need data to analyze.",
                    "evidence": [],
                    "query_analysis": {},
                }
                st.session_state.chat_messages.append(response_msg)
            else:
                # Build conversation history for the LLM
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages[:-1]  # exclude the current prompt
                ]

                with st.spinner("🧠 Retrieving & reasoning..."):
                    # Route through Rasa or ChatSense AI based on toggle
                    if st.session_state.use_rasa:
                        result = rasa_reason(
                            query=prompt,
                            engine=engine,
                            conversation_history=history,
                            stream=False,
                        )
                    else:
                        result = reason(
                            query=prompt,
                            engine=engine,
                            conversation_history=history,
                            stream=False,  # non-streaming for stability
                        )

                response_msg = {
                    "role": "assistant",
                    "content": result.get('response', 'No response generated.'),
                    "evidence": result.get('evidence', []),
                    "query_analysis": result.get('query_analysis', {}),
                }
                st.session_state.chat_messages.append(response_msg)

            st.rerun()

        # Clear button
        if len(st.session_state.chat_messages) > 1:
            if st.button("🗑️ Clear Chat", key="clear_ai"):
                st.session_state.chat_messages = [render_welcome_message()]
                st.rerun()
