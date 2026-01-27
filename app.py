import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Botpress URL
BOTPRESS_URL = "https://cdn.botpress.cloud/webchat/v3.5/shareable.html?configUrl=https://files.bpcontent.cloud/2025/04/05/17/20250405174729-I56NVWYS.json"

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'show_botpress' not in st.session_state:
    st.session_state.show_botpress = True

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

# Botpress AI Bot Panel
if st.session_state.show_botpress and bot_col is not None:
    with bot_col:
        st.markdown("### 🤖 AI Assistant")
        st.caption("Chat with our AI bot!")
        
        # Embed Botpress as iframe
        st.components.v1.iframe(
            BOTPRESS_URL,
            height=600,
            scrolling=True
        )
