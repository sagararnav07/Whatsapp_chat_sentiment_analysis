import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import requests
from groq import Groq
import json
import os

# API Configuration - Uses Streamlit secrets in production, env vars as fallback
def get_secret(key, default=""):
    """Get secret from Streamlit secrets or environment variable"""
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except:
        return os.environ.get(key, default)

# Groq API Configuration
GROQ_API_KEY = get_secret("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Botpress API Configuration
BOTPRESS_PAT = get_secret("BOTPRESS_PAT")
BOTPRESS_BOT_ID = get_secret("BOTPRESS_BOT_ID", "d84689bc-4df4-46cb-8c95-a78f0f51972f")
BOTPRESS_KB_ID = get_secret("BOTPRESS_KB_ID", "kb-2f0a7ea639")

def upload_chat_to_botpress_kb(chat_content: str, filename: str = "whatsapp-chat-data.txt"):
    """Upload chat data to Botpress Knowledge Base using 2-step process"""
    if not BOTPRESS_PAT or not BOTPRESS_BOT_ID or not BOTPRESS_KB_ID:
        return {"success": False, "error": "Botpress credentials not configured"}
    
    try:
        headers = {
            "Authorization": f"Bearer {BOTPRESS_PAT}",
            "x-bot-id": BOTPRESS_BOT_ID,
            "Content-Type": "application/json"
        }
        
        # File key for the knowledge base
        file_key = f"{BOTPRESS_KB_ID}/{filename}"
        
        # Convert content to bytes to get accurate size
        content_bytes = chat_content.encode('utf-8')
        file_size = len(content_bytes)
        
        # Step 1: Create the file entry (specifying size)
        create_url = "https://api.botpress.cloud/v1/files"
        payload = {
            "key": file_key,
            "size": file_size,
            "index": True,  # Index for semantic search
            "tags": {
                "source": "knowledge-base",
                "kbId": BOTPRESS_KB_ID,
                "title": "WhatsApp Chat Analysis Data"
            }
        }
        
        response = requests.put(create_url, headers=headers, json=payload)
        
        if response.status_code not in [200, 201]:
            return {"success": False, "error": f"Create file failed: {response.status_code} - {response.text[:200]}"}
        
        result = response.json()
        upload_url = result.get('file', {}).get('uploadUrl')
        
        if not upload_url:
            return {"success": False, "error": "No upload URL received"}
        
        # Step 2: Upload the actual content
        upload_response = requests.put(upload_url, data=content_bytes)
        
        if upload_response.status_code in [200, 201]:
            return {"success": True, "message": "Chat data uploaded to Knowledge Base!"}
        else:
            return {"success": False, "error": f"Upload content failed: {upload_response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_kb_document(df, selected_user="Overall"):
    """Generate a formatted document for the Knowledge Base"""
    if df is None or len(df) == 0:
        return None
    
    try:
        # Get users
        users = df['user'].unique().tolist()
        if 'group_notification' in users:
            users.remove('group_notification')
        
        # Basic stats
        total_msgs = len(df)
        total_words = int(df['message'].apply(lambda x: len(str(x).split())).sum())
        
        # User message counts
        user_counts = df[df['user'] != 'group_notification']['user'].value_counts().to_dict()
        
        # Date range
        first_date = df['date'].min().strftime('%B %d, %Y')
        last_date = df['date'].max().strftime('%B %d, %Y')
        duration = int((df['date'].max() - df['date'].min()).days)
        
        # Peak hour
        hourly = df.groupby('hour').size()
        peak_hour = int(hourly.idxmax()) if len(hourly) > 0 else 0
        
        # Media count
        media_count = int(df[df['message'].str.contains('<Media omitted>', case=False, na=False)].shape[0])
        
        # Sentiment analysis
        try:
            sentiment_df = helper.sentiment_analysis("Overall", df)
            if sentiment_df is not None and len(sentiment_df) > 0:
                positive = len(sentiment_df[sentiment_df['sentiment'] == 'Positive'])
                negative = len(sentiment_df[sentiment_df['sentiment'] == 'Negative'])
                neutral = len(sentiment_df[sentiment_df['sentiment'] == 'Neutral'])
            else:
                positive, negative, neutral = 0, 0, 0
        except:
            positive, negative, neutral = 0, 0, 0
        
        # Top emojis
        try:
            emoji_df = helper.emoji_analysis("Overall", df)
            top_emojis = ", ".join(emoji_df.head(10)['Emoji'].tolist()) if len(emoji_df) > 0 else "None"
        except:
            top_emojis = "None"
        
        # Sample messages
        sample_msgs = df.tail(50)[['user', 'message']].values.tolist()
        sample_text = "\n".join([f"- {m[0]}: {str(m[1])[:100]}" for m in sample_msgs])
        
        # Build document
        document = f"""# WhatsApp Chat Analysis Data

## Overview
This document contains the analysis of an uploaded WhatsApp chat. Use this data to answer user questions about their conversation.

## Chat Statistics
- **Total Messages**: {total_msgs:,}
- **Total Words**: {total_words:,}
- **Media Shared**: {media_count}
- **Date Range**: {first_date} to {last_date}
- **Duration**: {duration} days
- **Peak Activity Hour**: {peak_hour}:00

## Participants
{chr(10).join([f"- {user}" for user in users])}

## Message Counts by User
{chr(10).join([f"- {user}: {count:,} messages" for user, count in user_counts.items()])}

## Sentiment Analysis
- **Positive Messages**: {positive:,}
- **Neutral Messages**: {neutral:,}
- **Negative Messages**: {negative:,}
- **Overall Tone**: {"Positive" if positive > negative else "Negative" if negative > positive else "Neutral"}

## Top Emojis Used
{top_emojis}

## Recent Message Samples
{sample_text}

## How to Use This Data
When users ask questions like:
- "How many messages?" → Answer: {total_msgs:,} messages
- "Who talks the most?" → Check the message counts above
- "What's the sentiment?" → {positive} positive, {neutral} neutral, {negative} negative
- "When are they most active?" → Peak hour is {peak_hour}:00
"""
        return document
    except Exception as e:
        return f"Error generating document: {str(e)}"

def generate_chat_summary_for_botpress(df, selected_user="Overall"):
    """Generate a chat summary string to pass to Botpress"""
    if df is None or len(df) == 0:
        return json.dumps({"hasData": False, "message": "No chat uploaded yet"})
    
    try:
        # Get users
        users = df['user'].unique().tolist()
        if 'group_notification' in users:
            users.remove('group_notification')
        
        # Basic stats
        total_msgs = len(df)
        total_words = int(df['message'].apply(lambda x: len(str(x).split())).sum())
        
        # User message counts
        user_counts = df[df['user'] != 'group_notification']['user'].value_counts().head(10).to_dict()
        user_counts = {str(k): int(v) for k, v in user_counts.items()}
        
        # Date range
        first_date = df['date'].min().strftime('%B %d, %Y')
        last_date = df['date'].max().strftime('%B %d, %Y')
        duration = int((df['date'].max() - df['date'].min()).days)
        
        # Peak hour
        hourly = df.groupby('hour').size()
        peak_hour = int(hourly.idxmax()) if len(hourly) > 0 else 0
        
        # Media count
        media_count = int(df[df['message'].str.contains('<Media omitted>', case=False, na=False)].shape[0])
        
        # Sample recent messages (last 20 for context)
        sample_msgs = df.tail(20)[['user', 'message']].values.tolist()
        sample_text = " | ".join([f"{m[0]}: {str(m[1])[:50]}" for m in sample_msgs])
        
        # Sentiment summary
        try:
            sentiment_df = helper.sentiment_analysis("Overall", df)
            if sentiment_df is not None and len(sentiment_df) > 0:
                positive = int(len(sentiment_df[sentiment_df['sentiment'] == 'Positive']))
                negative = int(len(sentiment_df[sentiment_df['sentiment'] == 'Negative']))
                neutral = int(len(sentiment_df[sentiment_df['sentiment'] == 'Neutral']))
                sentiment = {"positive": positive, "neutral": neutral, "negative": negative}
            else:
                sentiment = {"positive": 0, "neutral": 0, "negative": 0}
        except:
            sentiment = {"positive": 0, "neutral": 0, "negative": 0}
        
        # Top emojis
        try:
            emoji_df = helper.emoji_analysis("Overall", df)
            top_emojis = emoji_df.head(5)['Emoji'].tolist() if len(emoji_df) > 0 else []
        except:
            top_emojis = []
        
        summary = {
            "hasData": True,
            "participants": users[:10],  # Limit to 10
            "totalMessages": total_msgs,
            "totalWords": total_words,
            "mediaShared": media_count,
            "dateRange": f"{first_date} to {last_date}",
            "duration": f"{duration} days",
            "peakHour": f"{peak_hour}:00",
            "selectedUser": selected_user,
            "userMessageCounts": user_counts,
            "sentiment": sentiment,
            "topEmojis": top_emojis,
            "recentMessages": sample_text[:500]  # Limit length
        }
        
        return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"hasData": False, "error": str(e)})

# Page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Intelligent AI Chat using Groq (Llama 3.1)
def get_bot_response(user_message, df=None, selected_user="Overall"):
    """Get intelligent AI response using Groq API"""
    
    # Build context about the chat data
    chat_context = ""
    if df is not None and len(df) > 0:
        try:
            # Get users
            users = df['user'].unique().tolist()
            if 'group_notification' in users:
                users.remove('group_notification')
            
            # Basic stats
            total_msgs = len(df)
            total_words = df['message'].apply(lambda x: len(str(x).split())).sum()
            
            # User message counts
            user_counts = df[df['user'] != 'group_notification']['user'].value_counts().head(10).to_dict()
            
            # Date range
            first_date = df['date'].min().strftime('%B %d, %Y')
            last_date = df['date'].max().strftime('%B %d, %Y')
            duration = (df['date'].max() - df['date'].min()).days
            
            # Peak hour
            hourly = df.groupby('hour').size()
            peak_hour = hourly.idxmax() if len(hourly) > 0 else 0
            
            # Media count
            media_count = df[df['message'].str.contains('<Media omitted>', case=False, na=False)].shape[0]
            
            # Sample recent messages (last 50 for context)
            sample_msgs = df.tail(100)[['user', 'message']].values.tolist()
            sample_text = "\n".join([f"{m[0]}: {m[1][:100]}" for m in sample_msgs[-30:]])
            
            # Emoji analysis
            try:
                emoji_df = helper.emoji_analysis("Overall", df)
                top_emojis = emoji_df.head(5).to_dict('records') if len(emoji_df) > 0 else []
            except:
                top_emojis = []
            
            # Sentiment analysis
            try:
                sentiment_df = helper.sentiment_analysis("Overall", df)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    positive = len(sentiment_df[sentiment_df['sentiment'] == 'Positive'])
                    negative = len(sentiment_df[sentiment_df['sentiment'] == 'Negative'])
                    neutral = len(sentiment_df[sentiment_df['sentiment'] == 'Neutral'])
                    sentiment_info = f"Positive: {positive}, Neutral: {neutral}, Negative: {negative}"
                else:
                    sentiment_info = "Not analyzed"
            except:
                sentiment_info = "Not analyzed"
            
            chat_context = f"""
## UPLOADED CHAT DATA:
- **Participants**: {', '.join(users)}
- **Total Messages**: {total_msgs:,}
- **Total Words**: {total_words:,}
- **Media Shared**: {media_count}
- **Date Range**: {first_date} to {last_date} ({duration} days)
- **Peak Activity Hour**: {peak_hour}:00
- **Currently Viewing**: {selected_user}

## MESSAGE COUNTS BY USER:
{chr(10).join([f"- {user}: {count} messages" for user, count in user_counts.items()])}

## TOP EMOJIS:
{chr(10).join([f"- {e['Emoji']}: {e['Count']} times" for e in top_emojis]) if top_emojis else "No emoji data"}

## SENTIMENT ANALYSIS:
{sentiment_info}

## RECENT MESSAGES SAMPLE:
{sample_text}
"""
        except Exception as e:
            chat_context = f"Chat data available but error extracting details: {str(e)}"
    else:
        chat_context = "No chat has been uploaded yet."
    
    # System prompt
    system_prompt = f"""You are an intelligent AI assistant for a WhatsApp Chat Sentiment Analysis app. You help users understand their WhatsApp conversations and the app's features.

{chat_context}

## YOUR CAPABILITIES:
1. Answer questions about the uploaded chat data (participants, message counts, activity patterns, etc.)
2. Explain app features (sentiment analysis, emoji tracking, activity patterns, word clouds)
3. Help users export WhatsApp chats (Android and iPhone instructions)
4. Provide insights about conversation patterns
5. Compare users' activity and messaging styles

## GUIDELINES:
- Be friendly and use emojis occasionally 😊
- Give specific numbers and data when available
- Keep responses concise but informative
- If asked about specific messages or conversations, use the sample messages provided
- If no chat is uploaded, guide users on how to upload one
- For export instructions: Android (⋮ → More → Export) and iPhone (Contact name → Export Chat)

## APP FEATURES TO EXPLAIN:
- 📊 Overview: Total messages, words, media, links, daily trends
- 💭 Sentiment: Positive/Negative/Neutral message breakdown using TextBlob
- 😀 Emojis: Top emojis used, per-user emoji analysis
- ⏰ Activity: 24-hour patterns, weekly heatmap, response times, night owl detection
- 🔍 Deep Dive: Word clouds, common words, chat streaks"""

    try:
        if not groq_client:
            return "❌ AI is not configured. Please set up the GROQ_API_KEY in secrets."
        
        # Call Groq API with Llama 3.3 (latest model)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {str(e)}\n\nPlease try again or check your internet connection."

# Custom CSS - Dark Theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    [data-testid="metric-container"] {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
    }
    
    [data-testid="stMetricValue"] {
        color: #58a6ff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d;
        border-radius: 8px;
        color: #8b949e;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #238636;
        color: white !important;
    }
    
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #2ea043;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'show_botpress' not in st.session_state:
    st.session_state.show_botpress = True
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI assistant powered by **Llama 3.3**! 🦙\n\nI can answer any questions about:\n• 📊 Your uploaded WhatsApp chat data\n• 📱 How to export and use this app\n• 💭 Sentiment analysis, emoji patterns, activity insights\n\nUpload a chat and ask me anything! 😊"}
    ]

# Sidebar
with st.sidebar:
    st.title("💬 WhatsApp Analyzer")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload Chat Export", type=['txt'])
    
    st.markdown("---")
    st.markdown("**How to export:**")
    st.markdown("1. Open WhatsApp chat")
    st.markdown("2. Tap ⋮ → More → Export")
    st.markdown("3. Choose 'Without Media'")
    st.markdown("4. Upload the .txt file")
    
    st.markdown("---")
    
    # Botpress AI Bot Toggle
    st.markdown("### 🤖 AI Assistant")
    st.session_state.show_botpress = st.toggle("Show AI Bot", value=st.session_state.show_botpress, help="Chat with our AI assistant")
    
    # Manual KB sync button (if credentials configured)
    if BOTPRESS_PAT and BOTPRESS_BOT_ID and BOTPRESS_KB_ID:
        if st.session_state.df is not None:
            if st.button("🔄 Sync Chat with AI", use_container_width=True):
                kb_doc = generate_kb_document(st.session_state.df, st.session_state.get('selected_user', 'Overall'))
                if kb_doc:
                    with st.spinner("📤 Uploading to Knowledge Base..."):
                        result = upload_chat_to_botpress_kb(kb_doc)
                        if result["success"]:
                            st.session_state.kb_uploaded = True
                            st.success("✅ Synced!")
                        else:
                            st.error(f"❌ {result.get('error', 'Failed')}")
            if st.session_state.get('kb_uploaded'):
                st.caption("✅ Chat data synced with Botpress KB")
    else:
        st.caption("⚙️ Configure Botpress API in app.py for KB sync")

# Create main layout with Botpress panel
if st.session_state.show_botpress:
    main_col, bot_col = st.columns([2, 1])
else:
    main_col = st.container()
    bot_col = None

# Process uploaded file
df = None
summary = None

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    summary = helper.get_chat_summary(df)
    
    st.session_state.df = df
    st.session_state.summary = summary
    
    # Auto-upload to Botpress Knowledge Base if configured
    if BOTPRESS_PAT and BOTPRESS_BOT_ID and BOTPRESS_KB_ID:
        if 'kb_uploaded' not in st.session_state or not st.session_state.kb_uploaded:
            kb_doc = generate_kb_document(df)
            if kb_doc:
                with st.spinner("📤 Syncing with AI Knowledge Base..."):
                    result = upload_chat_to_botpress_kb(kb_doc)
                    if result["success"]:
                        st.session_state.kb_uploaded = True
                        st.sidebar.success("✅ Chat synced with AI!")
                    else:
                        st.sidebar.warning(f"⚠️ KB sync failed: {result.get('error', 'Unknown')}")

# Main content area
with main_col:
    if uploaded_file is not None and df is not None:
        # Fetch unique users
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📊 Chat Analysis Dashboard")
            st.caption(f"📅 {summary['date_range']} • {summary['total_days']} days of conversation")
        with col2:
            selected_user = st.selectbox("👤 Select User", user_list)
        
        # Store selected user in session state for the bot
        st.session_state.selected_user = selected_user
        
        st.markdown("---")
        
        # Top Statistics
        st.subheader("📈 Quick Stats")
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("💬 Messages", f"{num_messages:,}")
        with col2:
            st.metric("📝 Words", f"{words:,}")
        with col3:
            st.metric("🖼️ Media", f"{num_media_messages:,}")
        with col4:
            st.metric("🔗 Links", f"{num_links:,}")
        with col5:
            st.metric("📅 Avg/Day", f"{summary['avg_messages_per_day']}")
        
        st.markdown("---")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "💭 Sentiment", "😀 Emojis", "⏰ Activity", "🔍 Deep Dive"
        ])
        
        # TAB 1: OVERVIEW
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📅 Monthly Message Trend")
                timeline = helper.monthly_timeline(selected_user, df)
                fig = px.area(timeline, x='time', y='message', 
                             color_discrete_sequence=['#58a6ff'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e',
                    xaxis_title="",
                    yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d')
                )
                fig.update_traces(fill='tozeroy', line=dict(width=2))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📆 Daily Activity")
                daily_timeline = helper.daily_timeline(selected_user, df)
                fig = px.line(daily_timeline, x='only_date', y='message',
                             color_discrete_sequence=['#3fb950'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e',
                    xaxis_title="",
                    yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Activity by Day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig = px.bar(x=busy_day.index, y=busy_day.values,
                            color=busy_day.values,
                            color_continuous_scale='Blues')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e',
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis_title="",
                    yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Activity by Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig = px.bar(x=busy_month.index, y=busy_month.values,
                            color=busy_month.values,
                            color_continuous_scale='Purples')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e',
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis_title="",
                    yaxis_title="Messages",
                    xaxis=dict(gridcolor='#21262d'),
                    yaxis=dict(gridcolor='#21262d')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if selected_user == 'Overall':
                st.markdown("#### 👥 Most Active Users")
                x, new_df = helper.most_busy_users(df)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(x=x.index, y=x.values, 
                                color=x.values,
                                color_continuous_scale='Greens',
                                labels={'x': 'User', 'y': 'Messages'})
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e',
                        showlegend=False,
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor='#21262d'),
                        yaxis=dict(gridcolor='#21262d')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(new_df, use_container_width=True, hide_index=True)
        
        # TAB 2: SENTIMENT ANALYSIS
        with tab2:
            st.subheader("💭 Sentiment Analysis")
            
            sentiment_counts, avg_sentiment, sentiment_df = helper.sentiment_analysis(selected_user, df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive = sentiment_counts.get('Positive', 0)
                total = sentiment_counts.sum()
                st.metric("😊 Positive", f"{positive:,}", 
                         delta=f"{positive/total*100:.1f}%" if total > 0 else "0%")
            with col2:
                neutral = sentiment_counts.get('Neutral', 0)
                st.metric("😐 Neutral", f"{neutral:,}",
                         delta=f"{neutral/total*100:.1f}%" if total > 0 else "0%")
            with col3:
                negative = sentiment_counts.get('Negative', 0)
                st.metric("😔 Negative", f"{negative:,}",
                         delta=f"{negative/total*100:.1f}%" if total > 0 else "0%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🥧 Sentiment Distribution")
                colors = {'Positive': '#3fb950', 'Neutral': '#d29922', 'Negative': '#f85149'}
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                            color=sentiment_counts.index,
                            color_discrete_map=colors,
                            hole=0.4)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if selected_user == 'Overall':
                    st.markdown("#### 👥 User Sentiment Comparison")
                    user_sentiment = helper.user_sentiment_comparison(df)
                    
                    colors = ['#3fb950' if x > 0 else '#f85149' for x in user_sentiment.values]
                    fig = go.Figure(go.Bar(
                        x=user_sentiment.values,
                        y=user_sentiment.index,
                        orientation='h',
                        marker_color=colors
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e',
                        xaxis_title="Sentiment Score",
                        yaxis_title="",
                        xaxis=dict(gridcolor='#21262d'),
                        yaxis=dict(gridcolor='#21262d')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            mood = "🌟 Very Positive" if avg_sentiment > 0.2 else "😊 Positive" if avg_sentiment > 0.05 else "😐 Neutral" if avg_sentiment > -0.05 else "😔 Slightly Negative" if avg_sentiment > -0.2 else "😢 Negative"
            st.info(f"**Overall Chat Mood:** {mood} (Score: {avg_sentiment:.3f})")
        
        # TAB 3: EMOJI ANALYSIS
        with tab3:
            st.subheader("😀 Emoji Analysis")
            
            emoji_df = helper.emoji_analysis(selected_user, df)
            
            if not emoji_df.empty:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### 🏆 Top Emojis")
                    for idx, row in emoji_df.head(10).iterrows():
                        st.markdown(f"**{row['Emoji']}** — {row['Count']} times")
                
                with col2:
                    st.markdown("#### 📊 Emoji Distribution")
                    fig = px.bar(emoji_df.head(15), x='Emoji', y='Count',
                                color='Count',
                                color_continuous_scale='Viridis')
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e',
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor='#21262d'),
                        yaxis=dict(gridcolor='#21262d')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emojis found!")
        
        # TAB 4: ACTIVITY ANALYSIS
        with tab4:
            st.subheader("⏰ Activity Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🕐 24-Hour Activity")
                hourly = helper.hourly_activity(selected_user, df)
                
                fig = go.Figure(go.Barpolar(
                    r=hourly.values,
                    theta=[f"{h}:00" for h in range(24)],
                    marker_color=hourly.values,
                    marker_colorscale='Blues',
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(showticklabels=False, ticks='', gridcolor='#21262d'),
                        angularaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d')
                    ),
                    font_color='#8b949e'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🗓️ Weekly Heatmap")
                user_heatmap = helper.activity_heatmap(selected_user, df)
                
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                user_heatmap = user_heatmap.reindex([d for d in day_order if d in user_heatmap.index])
                
                fig = px.imshow(user_heatmap,
                               color_continuous_scale='Blues',
                               aspect='auto')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if selected_user == 'Overall':
                st.markdown("#### ⚡ Response Times")
                response_times = helper.response_time_analysis(df)
                
                if not response_times.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(x=response_times.index, y=response_times.values,
                                    color=response_times.values,
                                    color_continuous_scale='RdYlGn_r')
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#8b949e',
                            coloraxis_showscale=False,
                            xaxis=dict(gridcolor='#21262d'),
                            yaxis=dict(gridcolor='#21262d')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fastest = response_times.idxmin()
                        slowest = response_times.idxmax()
                        st.success(f"⚡ **Fastest:** {fastest}")
                        st.warning(f"🐢 **Slowest:** {slowest}")
        
        # TAB 5: DEEP DIVE
        with tab5:
            st.subheader("🔍 Deep Dive Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ☁️ Word Cloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                if df_wc:
                    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0e1117')
                    ax.imshow(df_wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Not enough data")
            
            with col2:
                st.markdown("#### 📝 Common Words")
                most_common_df = helper.most_common_words(selected_user, df)
                
                if not most_common_df.empty:
                    fig = px.bar(most_common_df, x=1, y=0, orientation='h',
                                color=1, color_continuous_scale='Greens')
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#8b949e',
                        coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 Chat Streak")
                streak, start, end = helper.get_streak_info(df)
                st.metric("Consecutive Days", f"🔥 {streak} days")
                st.caption(f"{start} → {end}")
            
            with col2:
                st.markdown("#### 📅 Most Active Day")
                most_active_date, count = helper.get_most_active_date(selected_user, df)
                st.metric("Date", str(most_active_date))
                st.caption(f"{count} messages")
    
    else:
        # Welcome screen
        st.title("💬 WhatsApp Chat Analyzer")
        st.markdown("### Discover insights from your WhatsApp conversations")
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 📊")
            st.markdown("**Statistics**")
            st.caption("Message counts & media")
        
        with col2:
            st.markdown("### 💭")
            st.markdown("**Sentiment**")
            st.caption("Emotional analysis")
        
        with col3:
            st.markdown("### 😀")
            st.markdown("**Emojis**")
            st.caption("Usage patterns")
        
        with col4:
            st.markdown("### 🤖")
            st.markdown("**AI Bot**")
            st.caption("Ask anything!")
        
        st.markdown("---")
        st.info("👈 Upload your WhatsApp chat export from the sidebar to get started!")

# AI Chat Panel - Botpress Embedded + Llama Fallback
if st.session_state.show_botpress and bot_col is not None:
    with bot_col:
        st.markdown("### 🤖 AI Assistant")
        
        # Generate chat summary for Botpress
        chat_summary = generate_chat_summary_for_botpress(
            st.session_state.df, 
            st.session_state.get('selected_user', 'Overall')
        )
        # Escape for JavaScript
        chat_summary_escaped = chat_summary.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('"', '\\"')
        
        # Botpress Chat - Embedded directly with chat data
        botpress_html = f'''
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            #bp-webchat-container {{
                width: 100% !important;
                height: 520px !important;
                position: relative !important;
                border-radius: 16px;
                overflow: hidden;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            .bp-widget-widget, #bp-web-widget-container {{
                position: relative !important;
                width: 100% !important;
                height: 100% !important;
                bottom: 0 !important;
                right: 0 !important;
            }}
            .bpFab {{ display: none !important; }}
        </style>
        
        <div id="bp-webchat-container"></div>
        
        <script src="https://cdn.botpress.cloud/webchat/v3.5/inject.js"></script>
        <script>
            const chatDataJson = '{chat_summary_escaped}';
            let dataSent = false;
            
            window.botpress.on("ready", async function() {{
                window.botpress.open();
                
                // Update user with chat data
                try {{
                    const parsed = JSON.parse(chatDataJson);
                    await window.botpress.updateUser({{
                        name: "Chat Analyzer User",
                        data: {{
                            chatData: chatDataJson,
                            hasChat: parsed.hasData ? "yes" : "no",
                            msgCount: String(parsed.totalMessages || 0),
                            users: (parsed.participants || []).join(", "),
                            dates: parsed.dateRange || "none"
                        }}
                    }});
                    console.log("User data updated:", parsed);
                }} catch(e) {{
                    console.error("updateUser error:", e);
                }}
            }});
            
            window.botpress.on("webchat:ready", async function() {{
                if (dataSent) return;
                dataSent = true;
                
                // Send a custom event with the chat data
                try {{
                    const parsed = JSON.parse(chatDataJson);
                    if (parsed.hasData) {{
                        await window.botpress.sendEvent({{
                            type: "chatDataLoaded",
                            payload: {{
                                hasData: true,
                                totalMessages: parsed.totalMessages,
                                participants: parsed.participants,
                                dateRange: parsed.dateRange,
                                sentiment: parsed.sentiment,
                                topEmojis: parsed.topEmojis,
                                userCounts: parsed.userMessageCounts
                            }}
                        }});
                        console.log("Chat data event sent!");
                    }}
                }} catch(e) {{
                    console.error("sendEvent error:", e);
                }}
            }});
        </script>
        <script src="https://files.bpcontent.cloud/2025/04/05/17/20250405174729-R1ZMSM15.js?v={int(datetime.now().timestamp())}"></script>
        '''
        st.components.v1.html(botpress_html, height=540)
        
        # Show status indicator
        if st.session_state.df is not None:
            st.success("✅ Chat data synced with AI")
        else:
            st.info("📁 Upload a chat to enable data-aware responses")
        
        # Expander for Llama 3.3 (data-aware) as backup
        with st.expander("🦙 Llama 3.3 (Local Analysis)", expanded=False):
            if st.session_state.df is not None:
                st.success("✅ Chat loaded!")
            else:
                st.caption("📁 Upload a chat for data questions")
            
            # Mini chat container
            mini_chat = st.container(height=200)
            with mini_chat:
                for msg in st.session_state.chat_messages[-4:]:  # Show last 4 messages
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"])
            
            if prompt := st.chat_input("Ask Llama...", key="llama_input"):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                selected_user = st.session_state.get('selected_user', 'Overall')
                with st.spinner("🤔"):
                    response = get_bot_response(prompt, st.session_state.df, selected_user)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
