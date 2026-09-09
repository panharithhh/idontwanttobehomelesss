import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .registry import build_registry
from .schemas import TokenizeRequest, TokenizerResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tokenizer_api")

# Splits on whitespace and the zero-width space (U+200B), which Khmer text
# uses in place of spaces between words, to approximate a "word" count.
# Built via chr(0x200B) rather than a literal escape so the source file
# never has to contain an actual invisible character.
_ZWSP = chr(0x200B)
_WORD_SPLIT_RE = re.compile(r"[\s" + _ZWSP + r"]+")


def _word_count(text: str) -> int:
    return len([w for w in _WORD_SPLIT_RE.split(text) if w])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.registry = build_registry(settings)
    logger.info(
        "registry ready with %d tokenizer(s): %s",
        len(app.state.registry),
        list(app.state.registry),
    )
    yield
    app.state.registry.clear()


app = FastAPI(title="Khmer Tokenizer Fertility Benchmark API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/tokenizers")
async def list_tokenizers() -> list[str]:
    return list(app.state.registry)


@app.post("/tokenize")
async def tokenize(req: TokenizeRequest) -> dict[str, TokenizerResult]:
    settings = app.state.settings
    if len(req.text) > settings.max_input_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds max length of {settings.max_input_chars} characters",
        )

    word_count = max(_word_count(req.text), 1)  # avoid division by zero
    results: dict[str, TokenizerResult] = {}

    for name in req.tokenizers:
        adapter = app.state.registry.get(name)
        if adapter is None:
            raise HTTPException(status_code=422, detail=f"unknown tokenizer: {name!r}")
        tokens, ids = adapter.encode(req.text)
        results[name] = TokenizerResult(
            tokens=tokens,
            ids=ids,
            count=len(ids),
            fertility=len(ids) / word_count,
        )

    return results
