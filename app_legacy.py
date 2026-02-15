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
        # Try Streamlit secrets first (works in Streamlit Cloud)
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variable
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

# ============================================
# ADVANCED AI ENGINE - ChatSense AI v2.0
# ============================================

# Initialize semantic search model (cached for performance)
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for semantic search"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate
    except Exception as e:
        print(f"Embedding model not available: {e}")
        return None

# Cache embeddings for the current chat
@st.cache_data
def compute_message_embeddings(messages_text):
    """Compute embeddings for all messages"""
    model = load_embedding_model()
    if model is None or not messages_text:
        return None
    try:
        return model.encode(messages_text, show_progress_bar=False)
    except:
        return None

def semantic_search(query, df, top_k=10):
    """Find most relevant messages using semantic similarity"""
    model = load_embedding_model()
    if model is None or df is None or len(df) == 0:
        return []
    
    try:
        # Prepare messages
        messages = df[['user', 'message', 'date']].copy()
        messages['text'] = messages['user'] + ': ' + messages['message'].astype(str)
        messages_list = messages['text'].tolist()
        
        # Get embeddings (cached)
        embeddings = compute_message_embeddings(tuple(messages_list))
        if embeddings is None:
            return []
        
        # Encode query and find similar
        query_embedding = model.encode([query], show_progress_bar=False)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Relevance threshold
                results.append({
                    'user': messages.iloc[idx]['user'],
                    'message': messages.iloc[idx]['message'],
                    'date': messages.iloc[idx]['date'].strftime('%Y-%m-%d'),
                    'score': round(similarities[idx], 3)
                })
        return results
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []

def analyze_conversation_dynamics(df):
    """Deep analysis of conversation patterns"""
    if df is None or len(df) == 0:
        return {}
    
    try:
        users = [u for u in df['user'].unique() if u != 'group_notification']
        
        analysis = {
            'total_messages': len(df),
            'users': users,
            'user_stats': {},
            'conversation_starters': {},
            'response_patterns': {},
            'active_hours': {},
            'topics': []
        }
        
        # Per-user deep analysis
        for user in users:
            user_df = df[df['user'] == user]
            user_msgs = user_df['message'].astype(str)
            
            # Word count analysis
            word_counts = user_msgs.apply(lambda x: len(x.split()))
            
            # Question detection
            questions = user_msgs.str.contains(r'\?', regex=True).sum()
            
            # Media sharing
            media = user_msgs.str.contains('<Media omitted>', case=False).sum()
            
            # Link sharing
            links = user_msgs.str.contains(r'http[s]?://', regex=True).sum()
            
            # Emoji usage
            import emoji
            emoji_count = sum(len([c for c in str(msg) if c in emoji.EMOJI_DATA]) for msg in user_msgs)
            
            analysis['user_stats'][user] = {
                'messages': len(user_df),
                'words': int(word_counts.sum()),
                'avg_words_per_msg': round(word_counts.mean(), 1),
                'questions_asked': int(questions),
                'media_shared': int(media),
                'links_shared': int(links),
                'emojis_used': emoji_count,
                'percentage': round(len(user_df) / len(df) * 100, 1)
            }
        
        # Conversation starters (first message of the day)
        df_sorted = df.sort_values('date')
        daily_first = df_sorted.groupby(df_sorted['date'].dt.date).first()
        starter_counts = daily_first['user'].value_counts().to_dict()
        analysis['conversation_starters'] = starter_counts
        
        # Response time analysis (simplified)
        df_sorted = df.sort_values('date').reset_index(drop=True)
        if len(df_sorted) > 1:
            df_sorted['time_diff'] = df_sorted['date'].diff().dt.total_seconds() / 60  # in minutes
            df_sorted['prev_user'] = df_sorted['user'].shift(1)
            
            for user in users:
                responses = df_sorted[(df_sorted['user'] == user) & (df_sorted['prev_user'] != user)]
                if len(responses) > 0:
                    avg_response = responses['time_diff'].median()
                    analysis['response_patterns'][user] = {
                        'median_response_minutes': round(avg_response, 1) if avg_response < 1440 else 'varies'
                    }
        
        # Peak hours per user
        for user in users:
            user_hours = df[df['user'] == user]['hour'].value_counts().head(3).to_dict()
            analysis['active_hours'][user] = list(user_hours.keys())
        
        # Topic extraction (keyword-based)
        from collections import Counter
        all_text = ' '.join(df['message'].astype(str).str.lower())
        words = [w for w in all_text.split() if len(w) > 4 and not w.startswith('http')]
        
        # Filter common words
        stopwords = {'media', 'omitted', 'would', 'could', 'should', 'about', 'there', 'their', 'which', 'where', 'these', 'those', 'after', 'before', 'being', 'other'}
        words = [w for w in words if w not in stopwords]
        
        common_topics = Counter(words).most_common(10)
        analysis['topics'] = [w[0] for w in common_topics]
        
        return analysis
    except Exception as e:
        print(f"Analysis error: {e}")
        return {}

# Intelligent AI Chat using Groq (Llama 3.3) with RAG & Advanced Analytics
def get_bot_response(user_message, df=None, selected_user="Overall", conversation_history=None):
    """Get super-intelligent AI response using Groq API with RAG and deep analytics"""
    
    # ===== RAG: Semantic Search for Relevant Messages =====
    relevant_messages = ""
    if df is not None and len(df) > 0:
        search_results = semantic_search(user_message, df, top_k=8)
        if search_results:
            relevant_messages = "\n## 🔍 RELEVANT MESSAGES (Found via AI Search):\n"
            for r in search_results:
                relevant_messages += f"- [{r['date']}] {r['user']}: {r['message'][:200]}\n"
    
    # ===== Deep Analytics =====
    deep_analysis = ""
    if df is not None and len(df) > 0:
        analysis = analyze_conversation_dynamics(df)
        if analysis:
            deep_analysis = "\n## 📊 DEEP CONVERSATION ANALYSIS:\n"
            
            # User detailed stats
            if 'user_stats' in analysis:
                deep_analysis += "\n### Per-User Intelligence:\n"
                for user, stats in analysis['user_stats'].items():
                    deep_analysis += f"**{user}**:\n"
                    deep_analysis += f"  - Messages: {stats['messages']} ({stats['percentage']}%)\n"
                    deep_analysis += f"  - Words: {stats['words']:,} (avg {stats['avg_words_per_msg']} per msg)\n"
                    deep_analysis += f"  - Questions asked: {stats['questions_asked']}\n"
                    deep_analysis += f"  - Media shared: {stats['media_shared']}\n"
                    deep_analysis += f"  - Links shared: {stats['links_shared']}\n"
                    deep_analysis += f"  - Emojis used: {stats['emojis_used']}\n"
            
            # Conversation starters
            if 'conversation_starters' in analysis and analysis['conversation_starters']:
                deep_analysis += "\n### Who Starts Conversations:\n"
                for user, count in analysis['conversation_starters'].items():
                    deep_analysis += f"  - {user}: {count} days\n"
            
            # Response patterns
            if 'response_patterns' in analysis and analysis['response_patterns']:
                deep_analysis += "\n### Response Speed:\n"
                for user, pattern in analysis['response_patterns'].items():
                    deep_analysis += f"  - {user}: ~{pattern['median_response_minutes']} min median response\n"
            
            # Active hours
            if 'active_hours' in analysis and analysis['active_hours']:
                deep_analysis += "\n### Peak Active Hours:\n"
                for user, hours in analysis['active_hours'].items():
                    deep_analysis += f"  - {user}: {', '.join([f'{h}:00' for h in hours])}\n"
            
            # Topics
            if 'topics' in analysis and analysis['topics']:
                deep_analysis += f"\n### Main Topics/Keywords: {', '.join(analysis['topics'][:8])}\n"
    
    # Build context about the chat data
    chat_context = ""
    if df is not None and len(df) > 0:
        try:
            users = df['user'].unique().tolist()
            if 'group_notification' in users:
                users.remove('group_notification')
            
            total_msgs = len(df)
            total_words = df['message'].apply(lambda x: len(str(x).split())).sum()
            user_counts = df[df['user'] != 'group_notification']['user'].value_counts().head(10).to_dict()
            first_date = df['date'].min().strftime('%B %d, %Y')
            last_date = df['date'].max().strftime('%B %d, %Y')
            duration = (df['date'].max() - df['date'].min()).days
            hourly = df.groupby('hour').size()
            peak_hour = hourly.idxmax() if len(hourly) > 0 else 0
            media_count = df[df['message'].str.contains('<Media omitted>', case=False, na=False)].shape[0]
            
            # Sentiment
            try:
                sentiment_df = helper.sentiment_analysis("Overall", df)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    positive = len(sentiment_df[sentiment_df['sentiment'] == 'Positive'])
                    negative = len(sentiment_df[sentiment_df['sentiment'] == 'Negative'])
                    neutral = len(sentiment_df[sentiment_df['sentiment'] == 'Neutral'])
                    sentiment_info = f"Positive: {positive} ({round(positive/total_msgs*100,1)}%), Neutral: {neutral} ({round(neutral/total_msgs*100,1)}%), Negative: {negative} ({round(negative/total_msgs*100,1)}%)"
                else:
                    sentiment_info = "Not analyzed"
            except:
                sentiment_info = "Not analyzed"
            
            chat_context = f"""
## 📱 CHAT OVERVIEW:
- **Participants**: {', '.join(users)}
- **Total Messages**: {total_msgs:,}
- **Total Words**: {total_words:,}
- **Media Shared**: {media_count}
- **Date Range**: {first_date} to {last_date} ({duration} days)
- **Peak Hour**: {peak_hour}:00

## 📈 MESSAGE DISTRIBUTION:
{chr(10).join([f"- {user}: {count:,} messages ({round(count/total_msgs*100, 1)}%)" for user, count in user_counts.items()])}

## 💭 SENTIMENT BREAKDOWN:
{sentiment_info}
{deep_analysis}
{relevant_messages}
"""
        except Exception as e:
            chat_context = f"Chat data available but error extracting details: {str(e)}"
    else:
        chat_context = "No chat has been uploaded yet. Ask the user to upload a WhatsApp chat export."
    
    # ===== SUPER INTELLIGENT SYSTEM PROMPT =====
    system_prompt = f"""You are ChatSense AI v2.0 🧠 - an extraordinarily intelligent WhatsApp conversation analyst with PhD-level expertise in communication analysis, behavioral psychology, and data science.

{chat_context}

## 🎯 YOUR MISSION:
You don't just answer questions - you provide INSIGHTS that surprise and delight users. You notice patterns humans miss. You make connections that reveal relationship dynamics.

## 🧠 THINKING PROCESS (Use this for complex questions):
1. **Understand**: What is the user REALLY asking? What do they want to know?
2. **Analyze**: What data points are relevant? What patterns exist?
3. **Synthesize**: Connect multiple data points to form insights
4. **Deliver**: Present findings clearly with specific numbers

## 💡 INTELLIGENCE CAPABILITIES:
1. **Pattern Recognition** - Spot communication patterns, habits, rhythms
2. **Behavioral Analysis** - Who initiates? Who responds? Communication styles?
3. **Relationship Dynamics** - Balance of conversation, engagement levels
4. **Temporal Analysis** - When do they talk? Night owls? Morning people?
5. **Semantic Understanding** - What topics dominate? Emotional undertones?
6. **Predictive Insights** - What does the data suggest about the relationship?

## 🎨 RESPONSE STYLE:
- Start with a DIRECT answer, then expand with insights
- Use specific numbers and percentages
- Add emoji to make responses engaging but professional
- Provide unexpected insights ("Did you know...?")
- Be conversational yet authoritative
- For complex questions, show your reasoning briefly

## 📝 EXAMPLE RESPONSES:

**Q: "How many messages?"**
**A:** 📊 There are **289 messages** in this chat!

Here's the breakdown:
- Kartik Infy leads with **168 messages (58%)**
- Arnav sent **121 messages (42%)**

💡 *Insight: Kartik is 1.4x more active, but Arnav's messages tend to be longer. The conversation is fairly balanced for a friendship!*

**Q: "Tell me something interesting"**
**A:** 🔍 Here's something fascinating about this chat:

**The Night Owl Effect** 🦉
Kartik sends 23% of messages between 11 PM - 2 AM, while Arnav is mostly active during lunch (12-2 PM). You have complementary schedules!

**The Question Ratio** ❓
Arnav asks 2x more questions than Kartik, suggesting he often drives the conversation topics.

**Hidden Pattern** 🔮
Your longest chat streaks happen around weekends. Friendship maintenance mode activated on Saturdays!

## ⚠️ IMPORTANT RULES:
- ALWAYS use the actual data provided above
- If you find relevant messages via AI Search, reference them
- If asked about specific topics, search the relevant messages section
- Never make up statistics - use only what's in the data
- If something can't be determined, say so honestly

Remember: You're not just an assistant - you're a conversation ANALYST who reveals hidden patterns and provides genuine value!"""

    try:
        if not groq_client:
            return "❌ AI is not configured. Please set up the GROQ_API_KEY in secrets."
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for memory (last 8 exchanges)
        if conversation_history and len(conversation_history) > 1:
            recent_history = conversation_history[-9:-1] if len(conversation_history) > 9 else conversation_history[1:-1]
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"][:500]})
        
        messages.append({"role": "user", "content": user_message})
        
        # Call Groq API with Llama 3.3
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1500,  # More tokens for detailed responses
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {str(e)}\n\nPlease try again."

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
        {"role": "assistant", "content": "👋 Hey there! I'm **ChatSense AI v2.0** 🧠 - your **super-intelligent** WhatsApp analyst powered by advanced AI!\n\n**🚀 What makes me special:**\n• 🔬 **Semantic Search** - I find relevant messages using AI embeddings\n• 🧮 **Deep Analytics** - Response times, conversation patterns, topic detection\n• 🎯 **RAG-Powered** - I reference actual messages when answering\n• 💭 **Contextual Memory** - I remember our conversation\n• 🧠 **PhD-Level Analysis** - Psychological & sociological insights\n\n**💡 Try asking me:**\n• \"Who initiates conversations more?\"\n• \"What topics do we discuss most?\"\n• \"Analyze our communication patterns\"\n• \"What's the emotional dynamic of this chat?\"\n• \"When are response times slowest?\"\n\nUpload a chat to unlock my full potential! 🚀"}
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

# AI Chat Panel - ChatSense AI (Primary) + Botpress (Secondary)
if st.session_state.show_botpress and bot_col is not None:
    with bot_col:
        # PRIMARY: ChatSense AI (Llama 3.3)
        st.markdown("### 🤖 ChatSense AI")
        st.caption("Powered by Llama 3.3 70B • Remembers conversation • Analyzes your data")
        
        # Status indicator
        if st.session_state.df is not None:
            st.success("✅ Chat data loaded - I can answer specific questions!")
        else:
            st.info("📁 Upload a chat to unlock intelligent analysis")
        
        # Chat container - show more messages
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.chat_messages[-10:]:  # Show last 10 messages
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your chat...", key="llama_input"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            selected_user = st.session_state.get('selected_user', 'Overall')
            with st.spinner("🧠 Thinking..."):
                # Pass conversation history for memory
                response = get_bot_response(
                    prompt, 
                    st.session_state.df, 
                    selected_user,
                    conversation_history=st.session_state.chat_messages
                )
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Clear chat button
        if len(st.session_state.chat_messages) > 1:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # Keep welcome message
                st.rerun()
        
        st.markdown("---")
        
        # SECONDARY: Botpress in expander
        with st.expander("💬 Botpress Chat (Alternative)", expanded=False):
            st.caption("Alternative chatbot powered by Botpress")
            
            # Generate chat summary for Botpress
            chat_summary = generate_chat_summary_for_botpress(
                st.session_state.df, 
                st.session_state.get('selected_user', 'Overall')
            )
            # Escape for JavaScript
            chat_summary_escaped = chat_summary.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('"', '\\"')
            
            # Botpress Chat - Embedded
            botpress_html = f'''
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                #bp-webchat-container {{
                    width: 100% !important;
                    height: 450px !important;
                    position: relative !important;
                    border-radius: 12px;
                    overflow: hidden;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                }}
                .bp-widget-widget, #bp-web-widget-container {{
                    position: relative !important;
                    width: 100% !important;
                    height: 100% !important;
                }}
                .bpFab {{ display: none !important; }}
            </style>
            
            <div id="bp-webchat-container"></div>
            
            <script src="https://cdn.botpress.cloud/webchat/v3.5/inject.js"></script>
            <script>
                const chatDataJson = '{chat_summary_escaped}';
                window.botpress.on("ready", async function() {{
                    window.botpress.open();
                }});
            </script>
            <script src="https://files.bpcontent.cloud/2025/04/05/17/20250405174729-R1ZMSM15.js?v={int(datetime.now().timestamp())}"></script>
            '''
            st.components.v1.html(botpress_html, height=470)
