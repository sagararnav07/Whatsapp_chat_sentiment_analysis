# 💬 WhatsApp Chat Sentiment Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-blue?style=for-the-badge&logo=plotly&logoColor=white)
![AI](https://img.shields.io/badge/Groq-Llama_3.3-green?style=for-the-badge&logo=meta&logoColor=white)

**A powerful, AI-enhanced WhatsApp chat analyzer built with Streamlit**

[🚀 Live Demo](https://webchatanalyzer.streamlit.app/) · [Report Bug](https://github.com/sagararnav07/Whatsapp_chat_sentiment_analysis/issues) · [Request Feature](https://github.com/sagararnav07/Whatsapp_chat_sentiment_analysis/issues)

</div>

---

## ✨ Features

### 📊 **Comprehensive Analytics Dashboard**
- **Quick Stats** - Total messages, words, media shared, and links at a glance
- **User-wise Analysis** - Filter analytics by individual participants
- **Interactive Charts** - Beautiful Plotly visualizations

### 💭 **AI-Powered Sentiment Analysis**
- Real-time sentiment classification (Positive/Neutral/Negative)
- Per-user sentiment breakdown
- Sentiment distribution visualizations
- Powered by TextBlob NLP

### 🤖 **Integrated AI Chat Assistant**
- Ask questions about your chat in natural language
- Powered by **Groq Llama 3.3 70B** - blazing fast AI responses
- Botpress webchat integration for conversational experience
- Context-aware responses using your actual chat data

### 😀 **Emoji Analytics**
- Top emojis used across the conversation
- Per-user emoji preferences
- Emoji frequency charts and statistics

### ⏰ **Activity Pattern Analysis**
- **24-Hour Activity** - See when conversations peak
- **Weekly Heatmap** - Visualize chat patterns across days and hours
- **Response Time Analysis** - Average response times per user
- **Night Owl Detection** - Find out who's chatting late at night (12 AM - 5 AM)
- **Monthly Trends** - Track conversation volume over time

### 🔍 **Deep Dive Analytics**
- **Word Clouds** - Visual representation of most used words
- **Common Words** - Top 20 most frequently used words
- **Chat Streaks** - Longest consecutive days of chatting
- **First Message Analysis** - Who initiates conversations most

### 🌙 **Beautiful Dark Theme**
- Eye-friendly dark mode interface
- Gradient accents and modern UI design
- Responsive layout for all screen sizes

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sagararnav07/Whatsapp_chat_sentiment_analysis.git
   cd Whatsapp_chat_sentiment_analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up secrets** (for AI features)
   ```bash
   mkdir -p .streamlit
   cat > .streamlit/secrets.toml << EOF
   GROQ_API_KEY = "your-groq-api-key"
   BOTPRESS_PAT = "your-botpress-pat"
   BOTPRESS_BOT_ID = "your-bot-id"
   BOTPRESS_KB_ID = "your-kb-id"
   EOF
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

---

## 📱 How to Export WhatsApp Chats

### Android
1. Open the WhatsApp chat you want to analyze
2. Tap **⋮** (three dots) → **More** → **Export chat**
3. Select **Without media** (recommended for faster processing)
4. Save or share the `.txt` file

### iPhone
1. Open the WhatsApp chat
2. Tap the **contact/group name** at the top
3. Scroll down and tap **Export Chat**
4. Select **Without Media**
5. Save the `.txt` file

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation and analysis |
| **Plotly** | Interactive visualizations |
| **Matplotlib** | Static charts and word clouds |
| **TextBlob** | Sentiment analysis NLP |
| **Groq API** | AI chat (Llama 3.3 70B) |
| **Botpress** | Conversational AI webchat |
| **WordCloud** | Visual word frequency display |

---

## 📁 Project Structure

```
Whatsapp_chat_sentiment_analysis/
├── app.py                 # Main Streamlit application
├── preprocessor.py        # WhatsApp chat parsing logic
├── helper.py              # Analysis helper functions
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
├── setup.sh              # Server setup script
├── stop_hinglish.txt     # Custom stopwords for Hindi/English
├── .streamlit/
│   └── secrets.toml      # API keys (not in repo)
└── README.md             # This file
```

---

## 🔑 API Keys Setup

### Groq API (for AI Chat)
1. Go to [console.groq.com](https://console.groq.com)
2. Create an account and generate an API key
3. Add to `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

### Botpress (for Webchat - Optional)
1. Create a bot at [botpress.cloud](https://botpress.cloud)
2. Get your credentials from the dashboard
3. Add to `.streamlit/secrets.toml`:
   ```toml
   BOTPRESS_PAT = "bp_pat_..."
   BOTPRESS_BOT_ID = "your-bot-id"
   BOTPRESS_KB_ID = "your-kb-id"
   ```

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in the dashboard
5. Deploy!

### Heroku
```bash
heroku create your-app-name
git push heroku main
heroku config:set GROQ_API_KEY=your-key
```

---

## 📊 Sample Analytics

<details>
<summary>📈 Click to see sample outputs</summary>

### Quick Stats
| Metric | Value |
|--------|-------|
| Total Messages | 15,234 |
| Total Words | 89,567 |
| Media Shared | 423 |
| Links Shared | 156 |
| Duration | 365 days |

### Sentiment Distribution
- 🟢 Positive: 45%
- ⚪ Neutral: 40%
- 🔴 Negative: 15%

### Peak Activity
- Most Active Hour: 9 PM
- Most Active Day: Saturday
- Longest Streak: 45 days

</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) - For the amazing framework
- [Groq](https://groq.com) - For lightning-fast AI inference
- [Botpress](https://botpress.com) - For conversational AI platform
- [Plotly](https://plotly.com) - For beautiful interactive charts

---

<div align="center">

**Made with ❤️ by [Arnav Sagar](https://github.com/sagararnav07)**

⭐ Star this repo if you found it helpful!

</div>