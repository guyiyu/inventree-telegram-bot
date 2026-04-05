from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str
    gemini_api_key: str
    inventree_url: str = "http://inventree-server:8000"
    inventree_site_url: str = "https://inventree.example.com"
    inventree_api_token: str = ""
    allowed_user_ids: str = ""

    # Comma-separated model fallback lists. First model is tried first;
    # if rate-limited (429), the next model in the list is used.
    gemini_models_text: str = "gemini-flash-lite-latest"
    gemini_models_vision: str = "gemini-flash-latest,gemini-flash-lite-latest"

    # Conversation context budget (tokens). History is compacted to stay within this.
    context_budget: int = 32_000
    # Hot compaction: triggered in-chat when history exceeds this fraction of budget.
    # Keeps the last N messages raw for conversational continuity.
    hot_compaction_threshold: float = 0.9
    hot_compaction_keep_raw: int = 5
    # Idle compaction: triggered after idle_timeout seconds if history exceeds this
    # fraction of budget. Summarizes more aggressively, keeping fewer raw messages.
    idle_compaction_threshold: float = 0.7
    idle_compaction_keep_raw: int = 2
    idle_timeout: int = 300  # seconds (5 minutes)
    # Model used for compaction summaries (use the cheap one with 500 RPD)
    compaction_model: str = "gemini-flash-lite-latest"

    @property
    def allowed_users(self) -> set[int]:
        if not self.allowed_user_ids:
            return set()
        return {int(uid.strip()) for uid in self.allowed_user_ids.split(",") if uid.strip()}

    @property
    def text_models(self) -> list[str]:
        return [m.strip() for m in self.gemini_models_text.split(",") if m.strip()]

    @property
    def vision_models(self) -> list[str]:
        return [m.strip() for m in self.gemini_models_vision.split(",") if m.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
