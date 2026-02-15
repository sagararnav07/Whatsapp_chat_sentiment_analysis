"""
parser.py — Structured WhatsApp/Telegram Chat Parser
=====================================================
Parses raw .txt chat exports into a structured DataFrame with:
  - timestamp (datetime)
  - speaker (str)
  - message (str)

Handles:
  - WhatsApp bracket format: [DD/MM/YY, H:MM:SS AM/PM] Name: Message
  - WhatsApp dash format: DD/MM/YY, H:MM AM/PM - Name: Message
  - Multiline messages (appended to the previous entry)
  - Media/system notification filtering
"""

import re
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List


# ── Regex patterns for WhatsApp exports ──────────────────────────────────────
# Bracket format: [10/10/25, 7:32:49 AM] Username: message  (iOS / newer Android)
_BRACKET_RE = re.compile(
    r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s?[APap][Mm])\]\s'
)

# Dash format: 10/10/25, 7:32 AM - Username: message  (older Android)
_DASH_RE = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?[APap]?[Mm]?)\s-\s'
)

# Possible datetime parse formats, tried in order
_BRACKET_FMTS = [
    '%d/%m/%y, %I:%M:%S %p',
    '%d/%m/%Y, %I:%M:%S %p',
    '%m/%d/%y, %I:%M:%S %p',
    '%m/%d/%Y, %I:%M:%S %p',
    '%d/%m/%Y, %H:%M:%S',
    '%d/%m/%y, %H:%M:%S',
]

_DASH_FMTS = [
    '%d/%m/%Y, %H:%M',
    '%d/%m/%y, %H:%M',
    '%m/%d/%Y, %H:%M',
    '%m/%d/%y, %H:%M',
    '%d/%m/%Y, %I:%M %p',
    '%d/%m/%y, %I:%M %p',
    '%m/%d/%Y, %I:%M %p',
    '%m/%d/%y, %I:%M %p',
]

# Patterns that denote system/media messages (not real text)
MEDIA_PATTERNS = [
    'image omitted', 'video omitted', 'audio omitted',
    'sticker omitted', 'document omitted', 'Contact card omitted',
    'GIF omitted', '<Media omitted>', 'Missed voice call',
    'Missed video call', 'Voice call,', 'Video call,',
    'Messages and calls are end-to-end encrypted',
    'You deleted this message',
    'This message was deleted',
]


def is_media_or_system(message: str) -> bool:
    """Return True if the message is a media placeholder or system notification."""
    msg_lower = message.lower().strip()
    for pat in MEDIA_PATTERNS:
        if pat.lower() in msg_lower:
            return True
    return False


def parse_chat(raw_text: str) -> pd.DataFrame:
    """
    Parse a raw WhatsApp .txt export into a structured DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp : datetime64  — when the message was sent
        speaker   : str         — who sent it
        message   : str         — the message body
        is_media  : bool        — True if it's a media/system message
    """
    # Detect format
    if _BRACKET_RE.search(raw_text):
        pattern = _BRACKET_RE
        date_fmts = _BRACKET_FMTS
    else:
        pattern = _DASH_RE
        date_fmts = _DASH_FMTS

    # Split into (date_str, message_body) pairs
    parts = pattern.split(raw_text)
    # parts[0] is text before the first match – usually empty / preamble
    # Then alternating: date_str, message_body, date_str, message_body, …
    dates_raw = parts[1::2]
    bodies_raw = parts[2::2]

    if len(dates_raw) != len(bodies_raw):
        # If mismatch, trim to minimum
        n = min(len(dates_raw), len(bodies_raw))
        dates_raw = dates_raw[:n]
        bodies_raw = bodies_raw[:n]

    records: List[dict] = []

    for date_str, body in zip(dates_raw, bodies_raw):
        # Parse the timestamp
        ts = _parse_datetime(date_str.strip(), date_fmts)
        if ts is None:
            continue

        # Split "Speaker: message" — first colon separates name from message
        match = re.match(r'^([^:]+?):\s(.*)', body, re.DOTALL)
        if match:
            speaker = match.group(1).strip()
            message = match.group(2).strip()
        else:
            speaker = 'group_notification'
            message = body.strip()

        # Clean up invisible characters (zero-width, LRM, etc.)
        speaker = re.sub(r'[\u200e\u200f\u202a-\u202e\u200b\u2060\ufeff]', '', speaker).strip()
        message = re.sub(r'[\u200e\u200f\u202a-\u202e\u200b]', '', message).strip()

        records.append({
            'timestamp': ts,
            'speaker': speaker,
            'message': message,
            'is_media': is_media_or_system(message),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def _parse_datetime(date_str: str, fmts: list) -> Optional[datetime]:
    """Try multiple strptime formats, return first success or None."""
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    # Fallback: let pandas try
    try:
        return pd.to_datetime(date_str, dayfirst=True).to_pydatetime()
    except Exception:
        return None


def get_text_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Return only real human text messages (no media/system)."""
    if df.empty:
        return df
    filtered = df[(~df['is_media']) & (df['speaker'] != 'group_notification')].copy()
    # Also remove empty messages
    filtered = filtered[filtered['message'].str.strip().astype(bool)]
    return filtered.reset_index(drop=True)


def get_speakers(df: pd.DataFrame) -> List[str]:
    """Return list of unique speakers, excluding group_notification."""
    speakers = df[df['speaker'] != 'group_notification']['speaker'].unique().tolist()
    return sorted(speakers)


def filter_by_speaker(df: pd.DataFrame, speaker: str) -> pd.DataFrame:
    """Filter DataFrame to only messages from a specific speaker."""
    return df[df['speaker'] == speaker].reset_index(drop=True)


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter by date range.  Dates can be strings like '2025-03-15' or
    date objects.  Inclusive on both ends.
    """
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        start = pd.to_datetime(start_date)
        mask &= df['timestamp'] >= start
    if end_date is not None:
        end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask &= df['timestamp'] <= end
    return df[mask].reset_index(drop=True)


def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (first_date, last_date) as ISO strings."""
    return (
        df['timestamp'].min().strftime('%Y-%m-%d'),
        df['timestamp'].max().strftime('%Y-%m-%d'),
    )


def get_stats(df: pd.DataFrame, speaker: Optional[str] = None) -> dict:
    """
    Quick statistics about the parsed chat.
    Optionally filtered to a single speaker.
    """
    subset = df if speaker is None else filter_by_speaker(df, speaker)
    text_msgs = get_text_messages(subset)

    total_words = int(text_msgs['message'].apply(lambda x: len(x.split())).sum())
    media_count = int(subset['is_media'].sum())
    first_date = subset['timestamp'].min() if len(subset) > 0 else None
    last_date = subset['timestamp'].max() if len(subset) > 0 else None
    duration = (last_date - first_date).days if first_date and last_date else 0

    return {
        'total_messages': len(subset),
        'text_messages': len(text_msgs),
        'total_words': total_words,
        'media_count': media_count,
        'speakers': get_speakers(subset),
        'first_date': first_date.strftime('%B %d, %Y') if first_date else 'N/A',
        'last_date': last_date.strftime('%B %d, %Y') if last_date else 'N/A',
        'duration_days': duration,
    }
