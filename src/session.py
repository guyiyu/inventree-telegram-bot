"""Per-user conversation sessions with automatic history compaction.

Tracks message history for each Telegram user and compacts it when the
estimated token count approaches the configured budget. Two strategies:

- Hot compaction: triggered in-chat at 90% budget. Keeps last 5 messages raw.
- Idle compaction: triggered after 5 min idle at 70% budget. Keeps last 2 raw.

Both use the lite model (500 RPD) to generate a summary, preserving pending
actions and key decisions. Summaries are:
- Stored separately from messages (not injected as a fake user message)
- Injected into the system prompt as conversation context

Session persistence:
- Full session state (messages + summary) saved to data/sessions/{user_id}.json
  after every bot response and after compaction.
- On startup, sessions are restored from disk so conversations survive restarts.
- Inline image data is stripped from persisted messages to keep files small.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from config import settings
from compaction import refresh_context

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"

COMPACTION_PROMPT = """\
Summarize the following conversation between a user and a home inventory assistant.
Be concise but preserve ALL of the following:
- Items created, modified, moved, or deleted (with IDs and names)
- Categories and locations referenced or created
- User preferences or decisions expressed
- ANY pending action awaiting user confirmation (mark clearly as "PENDING: ...")
- Key context the assistant would need to continue helping

Output only the summary, no preamble."""


def _summary_path(user_id: int) -> Path:
    """Path to the persisted summary file for a user."""
    return SESSIONS_DIR / f"{user_id}.txt"


def _session_path(user_id: int) -> Path:
    """Path to the persisted session file (messages + summary) for a user."""
    return SESSIONS_DIR / f"{user_id}.json"


def _load_summary(user_id: int) -> str:
    """Load a persisted summary from disk, or return empty string."""
    path = _summary_path(user_id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _save_summary(user_id: int, summary: str) -> None:
    """Persist a summary to disk."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _summary_path(user_id).write_text(summary, encoding="utf-8")


def estimate_tokens(content: types.Content) -> int:
    """Estimate token count for a Gemini Content message.

    Uses a simple heuristic: ~4 chars per token for text, fixed estimate for
    images and function call/response overhead. This avoids burning an API
    call on countTokens.
    """
    tokens = 0
    for part in content.parts:
        if part.text:
            tokens += len(part.text) // 4
        elif part.inline_data:
            # Images are typically 258 tokens in Gemini
            tokens += 258
        elif part.function_call:
            # Function name + serialized args
            name_tokens = len(part.function_call.name) // 4
            args_tokens = len(str(part.function_call.args)) // 4 if part.function_call.args else 0
            tokens += name_tokens + args_tokens + 10  # overhead
        elif part.function_response:
            name_tokens = len(part.function_response.name) // 4
            resp_tokens = len(str(part.function_response.response)) // 4 if part.function_response.response else 0
            tokens += name_tokens + resp_tokens + 10
    # Role overhead
    tokens += 4
    return tokens


@dataclass
class ConversationSession:
    """Holds conversation history for a single user."""

    user_id: int
    messages: list[types.Content] = field(default_factory=list)
    summary: str = ""
    last_activity: float = field(default_factory=time.time)
    _token_estimate: int = 0

    def __post_init__(self):
        # Load persisted session from disk if available
        self._load_from_disk()

    @property
    def token_estimate(self) -> int:
        return self._token_estimate

    def add_message(self, content: types.Content) -> None:
        """Append a message and update the token estimate."""
        self.messages.append(content)
        self._token_estimate += estimate_tokens(content)
        self.last_activity = time.time()

    def add_messages(self, contents: list[types.Content]) -> None:
        """Append multiple messages (e.g. model response + function results)."""
        for content in contents:
            self.add_message(content)

    def clear(self) -> None:
        """Reset the session (messages and summary)."""
        self.messages.clear()
        self.summary = ""
        self._token_estimate = 0
        self.last_activity = time.time()
        # Remove persisted files
        for path in [_summary_path(self.user_id), _session_path(self.user_id)]:
            if path.exists():
                path.unlink()

    def save_to_disk(self) -> None:
        """Persist the full session (summary + messages) to disk as JSON.

        Inline image data is stripped and replaced with a text placeholder
        to avoid bloating the session file.
        """
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        serialized_messages = []
        for msg in self.messages:
            dumped = msg.model_dump(mode="json", exclude_none=True)
            # Strip inline_data (images) to keep the file small
            if "parts" in dumped:
                for part in dumped["parts"]:
                    if "inline_data" in part:
                        del part["inline_data"]
                        part["text"] = "[sent an image]"
            serialized_messages.append(dumped)

        data = {
            "summary": self.summary,
            "messages": serialized_messages,
            "last_activity": self.last_activity,
        }
        path = _session_path(self.user_id)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.debug("Session %s saved to disk (%d messages)", self.user_id, len(self.messages))

        # Also keep the plain-text summary file in sync (used during compaction)
        if self.summary:
            _save_summary(self.user_id, self.summary)

    def _load_from_disk(self) -> None:
        """Load session state from disk on startup.

        Tries the JSON session file first (has messages + summary).
        Falls back to the legacy plain-text summary file.
        """
        json_path = _session_path(self.user_id)
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self.summary = data.get("summary", "")
                self.last_activity = data.get("last_activity", time.time())
                for msg_data in data.get("messages", []):
                    content = types.Content.model_validate(msg_data)
                    self.messages.append(content)
                self._recalculate_tokens()
                logger.info(
                    "Session %s loaded from disk: %d messages, %d estimated tokens",
                    self.user_id, len(self.messages), self._token_estimate,
                )
                return
            except Exception:
                logger.exception("Failed to load session %s from JSON, starting fresh", self.user_id)
                self.messages.clear()
                self._token_estimate = 0

        # Fallback: legacy summary-only file
        txt_path = _summary_path(self.user_id)
        if txt_path.exists():
            self.summary = txt_path.read_text(encoding="utf-8")
            logger.info("Session %s loaded legacy summary from disk", self.user_id)

    def needs_hot_compaction(self) -> bool:
        threshold = int(settings.context_budget * settings.hot_compaction_threshold)
        return self._token_estimate >= threshold

    def needs_idle_compaction(self) -> bool:
        threshold = int(settings.context_budget * settings.idle_compaction_threshold)
        idle_seconds = time.time() - self.last_activity
        return (
            self._token_estimate >= threshold
            and idle_seconds >= settings.idle_timeout
        )

    async def compact(self, keep_raw: int) -> None:
        """Summarize old messages, keeping the last `keep_raw` messages raw.

        The summary is stored in self.summary (injected into the system prompt)
        and persisted to disk. Messages list is replaced with only recent raw
        messages.
        """
        if len(self.messages) <= keep_raw:
            logger.info("Session %s: not enough messages to compact (%d <= %d)",
                        self.user_id, len(self.messages), keep_raw)
            return

        old_messages = self.messages[:-keep_raw] if keep_raw > 0 else self.messages[:]
        recent_messages = self.messages[-keep_raw:] if keep_raw > 0 else []

        # Build the conversation text to summarize.
        # Include the existing summary as prior context so it accumulates.
        conversation_lines = []
        if self.summary:
            conversation_lines.append(f"[Prior summary]\n{self.summary}\n")
            conversation_lines.append("[New messages since last summary]")

        for msg in old_messages:
            role = msg.role or "unknown"
            for part in msg.parts:
                if part.text:
                    conversation_lines.append(f"{role}: {part.text}")
                elif part.function_call:
                    conversation_lines.append(
                        f"{role}: [called {part.function_call.name}({part.function_call.args})]"
                    )
                elif part.function_response:
                    # Truncate large function responses in the summary input
                    resp_str = str(part.function_response.response)
                    if len(resp_str) > 500:
                        resp_str = resp_str[:500] + "..."
                    conversation_lines.append(
                        f"{role}: [result of {part.function_response.name}: {resp_str}]"
                    )
                elif part.inline_data:
                    conversation_lines.append(f"{role}: [sent an image]")

        conversation_text = "\n".join(conversation_lines)

        # Ask the lite model to summarize
        try:
            response = _client.models.generate_content(
                model=settings.compaction_model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text=f"{COMPACTION_PROMPT}\n\n{conversation_text}"
                        )],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            summary_text = response.text or "(Failed to generate summary)"
        except ClientError:
            logger.exception("Compaction LLM call failed for user %s", self.user_id)
            # On failure, just trim oldest messages as a fallback
            half = len(self.messages) // 2
            self.messages = self.messages[half:]
            self._recalculate_tokens()
            self.save_to_disk()
            return

        logger.info(
            "Session %s compacted: %d messages → summary (%d chars) + %d raw",
            self.user_id, len(old_messages), len(summary_text), len(recent_messages),
        )

        # Store summary and persist full session to disk
        self.summary = summary_text

        # Keep only recent messages
        self.messages = recent_messages
        self._recalculate_tokens()

        self.save_to_disk()

        # Refresh inventory context so the next prompt has fresh data
        await refresh_context()

    def _recalculate_tokens(self) -> None:
        """Recalculate token estimate from scratch."""
        self._token_estimate = sum(estimate_tokens(m) for m in self.messages)


# Global session store, keyed by Telegram user_id
_sessions: dict[int, ConversationSession] = {}


def get_session(user_id: int) -> ConversationSession:
    """Get or create a conversation session for a user."""
    if user_id not in _sessions:
        _sessions[user_id] = ConversationSession(user_id=user_id)
    return _sessions[user_id]


def clear_session(user_id: int) -> None:
    """Clear a user's conversation history and summary."""
    if user_id in _sessions:
        _sessions[user_id].clear()


async def idle_compaction_loop() -> None:
    """Background loop that checks all sessions for idle compaction."""
    while True:
        await asyncio.sleep(60)  # check every minute
        for session in list(_sessions.values()):
            if session.needs_idle_compaction():
                logger.info(
                    "Idle compaction for user %s (%d tokens, idle %.0fs)",
                    session.user_id,
                    session.token_estimate,
                    time.time() - session.last_activity,
                )
                await session.compact(keep_raw=settings.idle_compaction_keep_raw)
