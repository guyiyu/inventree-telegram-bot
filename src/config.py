from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str
    gemini_api_key: str
    inventree_url: str = "http://inventree-server:8000"
    inventree_api_token: str = ""
    allowed_user_ids: str = ""

    @property
    def allowed_users(self) -> set[int]:
        if not self.allowed_user_ids:
            return set()
        return {int(uid.strip()) for uid in self.allowed_user_ids.split(",") if uid.strip()}

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
