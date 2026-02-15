"""
Rasa Analyser — Custom Actions
================================
These actions handle chat analysis queries by:
  1. Extracting entities (speaker, date) from the Rasa NLU parse.
  2. Calling back to the main app's data API for chat data.
  3. Formatting and returning the response.

The action server communicates with the main Streamlit app
via a lightweight REST API (see rasa_service.py for the bridge).
"""

import os
import json
import logging
from typing import Any, Text, Dict, List, Optional
from datetime import datetime

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# The data API endpoint — served by the main app or a sidecar
DATA_API_URL = os.environ.get("RASA_DATA_API_URL", "http://host.docker.internal:8501")


def _call_data_api(endpoint: str, payload: dict) -> Optional[dict]:
    """Call the main app's data API. Returns None on failure."""
    try:
        url = f"{DATA_API_URL}/{endpoint}"
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Data API call failed: {e}")
        return None


def _format_messages(messages: List[dict], max_show: int = 15) -> str:
    """Format a list of message dicts into a readable string."""
    if not messages:
        return "No messages found."

    lines = []
    for i, msg in enumerate(messages[:max_show]):
        date_str = msg.get("date_str", msg.get("timestamp", ""))
        speaker = msg.get("speaker", "Unknown")
        message = msg.get("message", "")
        media = " [MEDIA]" if msg.get("is_media", False) else ""
        lines.append(f"📌 [{date_str}] **{speaker}**: {message}{media}")

    result = "\n".join(lines)
    remaining = len(messages) - max_show
    if remaining > 0:
        result += f"\n\n... and {remaining} more messages."
    return result


def _extract_date(tracker: Tracker) -> Optional[str]:
    """Extract date entity from tracker."""
    date = tracker.get_slot("date")
    if date:
        return date
    # Try from latest entities
    for entity in tracker.latest_message.get("entities", []):
        if entity.get("entity") == "date":
            return entity.get("value")
    return None


def _extract_speaker(tracker: Tracker) -> Optional[str]:
    """Extract speaker entity from tracker."""
    speaker = tracker.get_slot("speaker")
    if speaker:
        return speaker
    for entity in tracker.latest_message.get("entities", []):
        if entity.get("entity") == "speaker":
            return entity.get("value")
    return None


# ── Analysis Actions ─────────────────────────────────────────────────────────

class ActionAnalyzeDate(Action):
    def name(self) -> Text:
        return "action_analyze_date"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        date = _extract_date(tracker)
        if not date:
            dispatcher.utter_message(response="utter_ask_date")
            return []

        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "date",
            "date": date,
            "speaker": None,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text=f"📅 Looking at **{date}**...\n\n"
                     f"I couldn't fetch detailed data right now. "
                     f"The data API might not be running. "
                     f"Try using the main ChatSense AI panel instead."
            )

        return [SlotSet("date", date)]


class ActionAnalyzeSpeaker(Action):
    def name(self) -> Text:
        return "action_analyze_speaker"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        speaker = _extract_speaker(tracker)
        if not speaker:
            dispatcher.utter_message(response="utter_ask_speaker")
            return []

        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "speaker",
            "date": None,
            "speaker": speaker,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text=f"👤 Looking for **{speaker}**'s messages...\n\n"
                     f"I couldn't fetch detailed data right now. "
                     f"Please ensure the chat file is uploaded in the main app."
            )

        return [SlotSet("speaker", speaker)]


class ActionAnalyzeDateSpeaker(Action):
    def name(self) -> Text:
        return "action_analyze_date_speaker"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        date = _extract_date(tracker)
        speaker = _extract_speaker(tracker)

        if not date:
            dispatcher.utter_message(response="utter_ask_date")
            return []
        if not speaker:
            dispatcher.utter_message(response="utter_ask_speaker")
            return []

        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "date_speaker",
            "date": date,
            "speaker": speaker,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text=f"📅👤 Looking for **{speaker}** on **{date}**...\n\n"
                     f"I couldn't fetch detailed data right now."
            )

        return [SlotSet("date", date), SlotSet("speaker", speaker)]


class ActionAnalyzeSummary(Action):
    def name(self) -> Text:
        return "action_analyze_summary"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "summary",
            "date": None,
            "speaker": None,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="📊 I'd love to summarize the chat, but I can't reach the data service right now."
            )
        return []


class ActionAnalyzeSentiment(Action):
    def name(self) -> Text:
        return "action_analyze_sentiment"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "sentiment",
            "date": None,
            "speaker": _extract_speaker(tracker),
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="💭 Sentiment analysis requires the data service to be running."
            )
        return []


class ActionAnalyzeDesire(Action):
    def name(self) -> Text:
        return "action_analyze_desire"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "desire",
            "date": None,
            "speaker": _extract_speaker(tracker),
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="💭 Desire detection requires the data service to be running."
            )
        return []


class ActionAnalyzeEmotions(Action):
    def name(self) -> Text:
        return "action_analyze_emotions"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "emotional_spike",
            "date": None,
            "speaker": None,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="⚡ Emotional spike detection requires the data service to be running."
            )
        return []


class ActionAnalyzeMedia(Action):
    def name(self) -> Text:
        return "action_analyze_media"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "media",
            "date": None,
            "speaker": None,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="🖼️ Media analysis requires the data service to be running."
            )
        return []


class ActionAnalyzeActivity(Action):
    def name(self) -> Text:
        return "action_analyze_activity"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "activity",
            "date": None,
            "speaker": None,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="⏰ Activity pattern analysis requires the data service to be running."
            )
        return []


class ActionAnalyzeGeneral(Action):
    def name(self) -> Text:
        return "action_analyze_general"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        result = _call_data_api("rasa/query", {
            "query": user_msg,
            "query_type": "general",
            "date": None,
            "speaker": None,
        })

        if result and result.get("response"):
            dispatcher.utter_message(text=result["response"])
        else:
            dispatcher.utter_message(
                text="I couldn't process that query right now. The data service might not be available."
            )
        return []
