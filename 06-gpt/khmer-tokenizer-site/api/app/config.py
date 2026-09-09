from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # comma-separated list of allowed CORS origins, e.g.
    # "http://localhost:5173,https://<user>.github.io"
    cors_origins: str = "http://localhost:5173"

    # local SentencePiece model file paths - never hardcoded, always from env
    sp_bpe_32k_model_path: str = ""
    sp_unigram_32k_model_path: str = ""

    # HuggingFace model ids for the transformers-based tokenizers.
    # xlm-roberta-base is a well-known, stable id. sea-lion/prahokbart ids
    # below are best-effort defaults - verify them against the current HF hub
    # before relying on them; a wrong id just means that loader fails and
    # gets excluded (see registry.py), it won't crash the app.
    xlmr_model_id: str = "xlm-roberta-base"
    sealion_model_id: str = "aisingapore/sea-lion-7b"
    prahokbart_model_id: str = ""

    max_input_chars: int = 2000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
