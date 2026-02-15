"""
rasa_service.py — Rasa Analyser Service Layer
===============================================
Bridge between the Streamlit app and the Rasa NLP engine.

Architecture:
  ┌─────────────┐     HTTP      ┌──────────────┐
  │  Streamlit   │ ──────────── │  Rasa Server  │
  │  (app.py)    │   REST API   │  (port 5005)  │
  └──────┬───────┘              └──────┬────────┘
         │                             │
         │  rasa_service.py            │  Custom Actions
         │  (this file)                │  (actions.py)
         │                             │
         ▼                             ▼
  ┌──────────────┐              ┌──────────────┐
  │  Retrieval   │ ◄────────── │ Action Server │
  │  Engine      │   fallback   │  (port 5055) │
  └──────────────┘              └──────────────┘

Two modes of operation:
  1. NLU-only:  Send to /model/parse → get intent + entities →
                feed into existing retrieval pipeline
  2. Full Rasa: Send to /webhooks/rest/webhook → get complete response

The mode is controlled by RASA_MODE environment variable.

Fallback: If Rasa is down or fails, falls back to the existing
ChatSense AI pipeline (retrieval.py + reasoning.py).
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

def _get_config() -> Dict[str, Any]:
    """Load Rasa configuration from environment variables."""
    return {
        'enabled': os.environ.get('USE_RASA', 'false').lower() == 'true',
        'url': os.environ.get('RASA_URL', 'http://localhost:5005'),
        'mode': os.environ.get('RASA_MODE', 'nlu_only'),  # 'nlu_only' or 'full'
        'timeout': int(os.environ.get('RASA_TIMEOUT', '10')),
        'sender_id': os.environ.get('RASA_SENDER_ID', 'streamlit_user'),
    }


def is_rasa_enabled() -> bool:
    """Check if Rasa integration is enabled via environment."""
    # Also check Streamlit secrets
    enabled = os.environ.get('USE_RASA', 'false').lower() == 'true'
    if not enabled:
        try:
            import streamlit as st
            if 'USE_RASA' in st.secrets:
                enabled = str(st.secrets['USE_RASA']).lower() == 'true'
        except Exception:
            pass
    return enabled


def is_rasa_healthy() -> bool:
    """Check if Rasa server is reachable and healthy."""
    config = _get_config()
    try:
        resp = requests.get(f"{config['url']}/", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ── NLU Parsing ──────────────────────────────────────────────────────────────

def parse_nlu(message: str) -> Optional[Dict[str, Any]]:
    """
    Send a message to Rasa's /model/parse endpoint for NLU-only processing.

    Returns parsed result with:
      - intent: {name, confidence}
      - entities: [{entity, value, start, end, confidence}]
      - text: original message

    Returns None if Rasa is unreachable.
    """
    config = _get_config()
    try:
        resp = requests.post(
            f"{config['url']}/model/parse",
            json={"text": message},
            timeout=config['timeout'],
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("Rasa server not reachable at %s", config['url'])
        return None
    except requests.exceptions.Timeout:
        logger.warning("Rasa server timed out")
        return None
    except Exception as e:
        logger.error("Rasa NLU parse failed: %s", e)
        return None


def nlu_to_query_analysis(
    nlu_result: Dict[str, Any],
    speakers: List[str],
) -> Dict[str, Any]:
    """
    Convert Rasa NLU output into the same query analysis format
    used by retrieval.py's analyze_query().

    This allows Rasa NLU to be a drop-in replacement for the
    regex-based query analyzer while keeping the rest of the
    pipeline (retrieval + reasoning) identical.
    """
    intent = nlu_result.get('intent', {})
    intent_name = intent.get('name', 'general')
    confidence = intent.get('confidence', 0.0)
    entities = nlu_result.get('entities', [])

    # Extract speaker entity
    speaker = None
    for ent in entities:
        if ent.get('entity') == 'speaker':
            candidate = ent.get('value', '').strip()
            # Fuzzy match against known speakers
            for sp in speakers:
                if candidate.lower() in sp.lower() or sp.lower() in candidate.lower():
                    speaker = sp
                    break
            if not speaker:
                speaker = candidate  # Use raw value if no match
            break

    # Extract date entity
    date_str = None
    for ent in entities:
        if ent.get('entity') == 'date':
            date_str = ent.get('value', '').strip()
            break

    # Parse date if found
    from retrieval import _parse_flexible_date
    parsed_date = _parse_flexible_date(date_str) if date_str else None

    # Map Rasa intent to query analysis format
    intent_map = {
        'ask_date_messages': 'date',
        'ask_speaker_messages': 'speaker',
        'ask_date_speaker_messages': 'date_speaker',
        'ask_summary': 'summary',
        'ask_sentiment': 'semantic',
        'ask_desire_detection': 'desire',
        'ask_emotional_spikes': 'emotional_spike',
        'ask_media_shared': 'semantic',
        'ask_activity_pattern': 'semantic',
        'ask_general': 'general',
        'greet': 'general',
        'goodbye': 'general',
        'ask_help': 'general',
        'out_of_scope': 'general',
        'nlu_fallback': 'semantic',
    }

    query_type = intent_map.get(intent_name, 'semantic')

    # Override query_type based on extracted entities
    if speaker and parsed_date:
        query_type = 'date_speaker'
    elif parsed_date and not speaker:
        query_type = 'date'
    elif speaker and not parsed_date:
        if query_type not in ('desire', 'emotional_spike', 'summary'):
            query_type = 'speaker'

    is_desire = intent_name == 'ask_desire_detection' or query_type == 'desire'

    return {
        'speaker': speaker,
        'start_date': parsed_date,
        'end_date': parsed_date,
        'is_desire_query': is_desire,
        'is_date_query': parsed_date is not None,
        'is_speaker_query': speaker is not None,
        'query_type': query_type,
        'cleaned_query': nlu_result.get('text', ''),
        # Rasa-specific metadata
        'rasa_intent': intent_name,
        'rasa_confidence': confidence,
        'rasa_entities': entities,
    }


# ── Full Rasa Interaction ────────────────────────────────────────────────────

def query_rasa(
    message: str,
    sender_id: str = "streamlit_user",
) -> Optional[Dict[str, Any]]:
    """
    Send a message to Rasa via the REST webhook (full conversation mode).

    Returns a dict with:
      - responses: list of bot response texts
      - raw: full Rasa response

    Returns None if Rasa is unreachable.
    """
    config = _get_config()
    try:
        resp = requests.post(
            f"{config['url']}/webhooks/rest/webhook",
            json={
                "sender": sender_id,
                "message": message,
            },
            timeout=config['timeout'],
        )
        resp.raise_for_status()
        data = resp.json()

        # Rasa returns a list of bot utterances
        responses = []
        for item in data:
            if 'text' in item:
                responses.append(item['text'])

        return {
            'responses': responses,
            'response': '\n\n'.join(responses) if responses else None,
            'raw': data,
        }
    except requests.exceptions.ConnectionError:
        logger.warning("Rasa server not reachable for webhook at %s", config['url'])
        return None
    except Exception as e:
        logger.error("Rasa webhook query failed: %s", e)
        return None


# ── Unified Query Function ───────────────────────────────────────────────────

def rasa_reason(
    query: str,
    engine: Any,  # RetrievalEngine
    conversation_history: Optional[List[Dict]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Rasa-powered reasoning pipeline.

    Two modes:
    1. NLU-only (default): Rasa does intent/entity extraction,
       then feeds into existing retrieval + Groq reasoning.
    2. Full: Rasa handles the entire conversation via webhook.

    Falls back to existing ChatSense AI pipeline on any failure.

    Returns same format as reasoning.py's reason():
      {'response': str, 'evidence': list, 'query_analysis': dict}
    """
    config = _get_config()

    if config['mode'] == 'full':
        # ── Full Rasa Mode ──
        result = query_rasa(query, config['sender_id'])
        if result and result.get('response'):
            return {
                'response': result['response'],
                'evidence': [],
                'query_analysis': {
                    'query_type': 'rasa_full',
                    'rasa_mode': 'full',
                },
            }
        # Fall through to fallback

    elif config['mode'] == 'nlu_only':
        # ── NLU-only Mode (recommended) ──
        nlu_result = parse_nlu(query)
        if nlu_result:
            speakers = engine.speakers if engine else []
            qa = nlu_to_query_analysis(nlu_result, speakers)

            # Handle simple intents that don't need retrieval
            simple_intents = {
                'greet': "👋 Hello! I'm **Rasa Analyser**. Ask me anything about the chat!",
                'goodbye': "Goodbye! Come back anytime. 👋",
                'ask_help': (
                    "I can help with:\n"
                    "- 📅 Date queries: \"What was said on 17/10/25?\"\n"
                    "- 👤 Speaker queries: \"What did Arnav say?\"\n"
                    "- 💭 Desire detection: \"Did anyone wish for something?\"\n"
                    "- ⚡ Emotional analysis: \"When did arguments happen?\"\n"
                    "- 📊 Summaries: \"Summarize the chat\""
                ),
                'out_of_scope': "I'm specialized in WhatsApp chat analysis. Could you ask about the chat data?",
            }

            intent_name = qa.get('rasa_intent', '')
            if intent_name in simple_intents:
                return {
                    'response': simple_intents[intent_name],
                    'evidence': [],
                    'query_analysis': qa,
                }

            # For analysis intents, use Rasa NLU + existing pipeline
            if engine:
                # Override the retrieval engine's query analysis with Rasa's
                from reasoning import reason as chatsense_reason
                try:
                    result = chatsense_reason(
                        query=query,
                        engine=engine,
                        conversation_history=conversation_history,
                        stream=stream,
                    )
                    # Enrich with Rasa metadata
                    result['query_analysis'] = {
                        **result.get('query_analysis', {}),
                        'rasa_intent': qa.get('rasa_intent'),
                        'rasa_confidence': qa.get('rasa_confidence'),
                        'rasa_mode': 'nlu_only',
                    }
                    return result
                except Exception as e:
                    logger.error("Rasa NLU + retrieval pipeline failed: %s", e)
                    # Fall through to fallback

    # ── Fallback to existing ChatSense AI ──
    logger.info("Falling back to ChatSense AI pipeline")
    try:
        from reasoning import reason as chatsense_reason
        result = chatsense_reason(
            query=query,
            engine=engine,
            conversation_history=conversation_history,
            stream=stream,
        )
        result['query_analysis'] = {
            **result.get('query_analysis', {}),
            'rasa_fallback': True,
        }
        return result
    except Exception as e:
        return {
            'response': f"❌ Both Rasa and ChatSense AI failed: {str(e)}",
            'evidence': [],
            'query_analysis': {'rasa_fallback': True, 'error': str(e)},
        }
