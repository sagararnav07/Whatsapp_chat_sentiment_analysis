# Rasa Analyser — README
# ========================
# Rasa NLP integration for the WhatsApp Chat Analyser.

## Overview

**Rasa Analyser** is an OPTIONAL NLP engine that runs alongside the existing
ChatSense AI chatbot. It provides intent classification and entity extraction
via the Rasa framework.

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit App (app.py)                 │
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────┐    │
│  │  USE_RASA=false  │    │     USE_RASA=true         │    │
│  │  (default)       │    │                            │    │
│  │                  │    │  ┌────────────────────┐    │    │
│  │  analyze_query() │    │  │  rasa_service.py   │    │    │
│  │  (regex-based)   │    │  │  ┌──────────────┐  │    │    │
│  │       │          │    │  │  │ parse_nlu()  │  │    │    │
│  │       ▼          │    │  │  └──────┬───────┘  │    │    │
│  │  RetrievalEngine │    │  │         ▼          │    │    │
│  │       │          │    │  │  Rasa Server:5005  │    │    │
│  │       ▼          │    │  │  (NLU parse)       │    │    │
│  │  reason()        │    │  │         │          │    │    │
│  │  (Groq LLM)     │    │  │         ▼          │    │    │
│  │       │          │    │  │  RetrievalEngine   │    │    │
│  │       ▼          │    │  │         │          │    │    │
│  │  Response        │    │  │         ▼          │    │    │
│  └─────────────────┘    │  │  reason() (Groq)   │    │    │
│                          │  │         │          │    │    │
│                          │  │         ▼          │    │    │
│                          │  │  Response          │    │    │
│                          │  └────────────────────┘    │    │
│                          └──────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the Rasa Server (Docker)

```bash
cd rasa/
docker-compose up --build
```

This starts:
- **Rasa Server** on `http://localhost:5005`
- **Action Server** on `http://localhost:5055`

### 2. Enable Rasa in the App

**Option A — Environment variable:**
```bash
USE_RASA=true streamlit run app.py
```

**Option B — Sidebar toggle:**
Toggle "Use Rasa NLP" in the sidebar at runtime.

**Option C — Streamlit secrets:**
Add to `.streamlit/secrets.toml`:
```toml
USE_RASA = "true"
```

### 3. Verify Connection

When Rasa is enabled and the server is running, you'll see:
- ✅ **Green badge** in the sidebar: "Rasa connected"

If the server is down:
- ⚠️ **Warning**: "Rasa server not reachable. Will fallback to ChatSense AI."

## Architecture

### File Structure

```
webchatAnalyzer/
├── app.py                  # Main app (minimal Rasa routing added)
├── rasa_service.py         # 🆕 Rasa service bridge layer
├── .env.example            # 🆕 Environment configuration template
│
├── rasa/                   # 🆕 Full Rasa project
│   ├── config.yml          # Pipeline & policies
│   ├── domain.yml          # Intents, entities, slots, responses
│   ├── credentials.yml     # Channel config (REST)
│   ├── endpoints.yml       # Action server endpoint
│   ├── Dockerfile          # Multi-stage Docker build
│   ├── docker-compose.yml  # Rasa + Action server compose
│   │
│   ├── data/
│   │   ├── nlu.yml         # NLU training examples
│   │   ├── stories.yml     # Conversation stories
│   │   └── rules.yml       # Deterministic rules
│   │
│   └── actions/
│       ├── __init__.py
│       └── actions.py      # Custom actions
│
├── parser.py               # ✅ Unchanged
├── embeddings.py           # ✅ Unchanged
├── retrieval.py            # ✅ Unchanged
├── reasoning.py            # ✅ Unchanged
├── ui.py                   # ✅ Unchanged
├── preprocessor.py         # ✅ Unchanged
├── helper.py               # ✅ Unchanged
└── ...
```

### Integration Modes

| Mode | Env Var | Behavior |
|------|---------|----------|
| **NLU-only** (recommended) | `RASA_MODE=nlu_only` | Rasa extracts intents & entities → existing retrieval + Groq |
| **Full Rasa** | `RASA_MODE=full` | Rasa handles complete conversation via webhook |

### Fallback Strategy

```
User Query
    │
    ▼
USE_RASA=true?
    │
    ├──No──→ ChatSense AI (existing pipeline)
    │
    ├──Yes──→ Rasa Server reachable?
    │             │
    │             ├──No──→ Fallback to ChatSense AI
    │             │
    │             └──Yes──→ Rasa processes query
    │                          │
    │                          └──Error?──→ Fallback to ChatSense AI
    │
    ▼
Response to User
```

## Training Rasa Locally (without Docker)

```bash
# Install Rasa
pip install rasa

# Train the model
cd rasa/
rasa train

# Start the server
rasa run --enable-api --cors "*" --port 5005

# In another terminal, start the action server
rasa run actions --port 5055
```

## Supported Intents

| Intent | Example Query |
|--------|---------------|
| `ask_date_messages` | "What was said on 15/11/25?" |
| `ask_speaker_messages` | "What did Arnav say?" |
| `ask_date_speaker_messages` | "What did Kartik say on 25/12/25?" |
| `ask_summary` | "Summarize the chat" |
| `ask_sentiment` | "What's the mood of the chat?" |
| `ask_desire_detection` | "Did anyone wish for something?" |
| `ask_emotional_spikes` | "When did arguments happen?" |
| `ask_media_shared` | "What media was shared?" |
| `ask_activity_pattern` | "When are they most active?" |
| `ask_general` | "How many messages are there?" |

## Notes

- The existing ChatSense AI chatbot is **completely unaffected** when Rasa is disabled.
- Rasa is a **compile-time optional** dependency — the app works without it installed.
- The Rasa toggle in the sidebar provides runtime switching between engines.
- All Rasa configuration is isolated in `/rasa/` and `rasa_service.py`.
