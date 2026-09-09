from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    tokenizers: list[str] = Field(..., min_length=1)


class TokenizerResult(BaseModel):
    tokens: list[str]
    ids: list[int]
    count: int
    fertility: float
