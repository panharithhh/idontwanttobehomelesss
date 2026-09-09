import type { TokenizeResponse } from './types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function fetchTokenizerNames(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/tokenizers`)
  if (!res.ok) throw new Error(`failed to load tokenizer list (${res.status})`)
  return res.json()
}

export async function tokenize(text: string, tokenizers: string[]): Promise<TokenizeResponse> {
  const res = await fetch(`${API_BASE}/tokenize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, tokenizers }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `request failed (${res.status})`)
  }
  return res.json()
}
