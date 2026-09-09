"""
Tokenizer registry: name -> loaded tokenizer adapter.

Populated once at app startup (see main.py's lifespan) via `build_registry()`.
Each loader is isolated in its own try/except so one broken or missing model
doesn't take down the whole app - it just gets excluded from the registry,
and a warning is logged so it's obvious which tokenizers are actually live.
"""

import logging
from typing import Callable, Protocol

from .config import Settings

logger = logging.getLogger("tokenizer_registry")


class TokenizerAdapter(Protocol):
    """ Common interface every registry entry exposes to the API layer. """

    def encode(self, text: str) -> tuple[list[str], list[int]]: ...


class SentencePieceAdapter:
    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("no model path configured")
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        ids = self._sp.encode(text, out_type=int)
        tokens = self._sp.encode(text, out_type=str)
        return tokens, ids


class HFAdapter:
    def __init__(self, model_id: str):
        if not model_id:
            raise ValueError("no model id configured")
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_id)

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        ids = self._tok.encode(text, add_special_tokens=False)
        tokens = self._tok.convert_ids_to_tokens(ids)
        return tokens, ids


class TiktokenAdapter:
    def __init__(self, encoding_name: str = "o200k_base"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        ids = self._enc.encode(text)
        tokens = [self._enc.decode([i]) for i in ids]
        return tokens, ids


def build_registry(settings: Settings) -> dict[str, TokenizerAdapter]:
    """ Try to load every known tokenizer. Anything that fails to load
        (missing file, missing package, network error fetching a HF model,
        ...) is logged and simply excluded, rather than crashing startup. """
    loaders: dict[str, Callable[[], TokenizerAdapter]] = {
        "sp-bpe-32k": lambda: SentencePieceAdapter(settings.sp_bpe_32k_model_path),
        "sp-unigram-32k": lambda: SentencePieceAdapter(settings.sp_unigram_32k_model_path),
        "xlm-r": lambda: HFAdapter(settings.xlmr_model_id),
        "sea-lion": lambda: HFAdapter(settings.sealion_model_id),
        "prahokbart": lambda: HFAdapter(settings.prahokbart_model_id),
        "gpt-4o": lambda: TiktokenAdapter("o200k_base"),
    }

    registry: dict[str, TokenizerAdapter] = {}
    for name, loader in loaders.items():
        try:
            registry[name] = loader()
            logger.info("loaded tokenizer: %s", name)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.warning("failed to load tokenizer %r, excluding it: %s", name, exc)
    return registry
