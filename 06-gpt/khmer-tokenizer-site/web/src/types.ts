export interface TokenizerResult {
  tokens: string[]
  ids: number[]
  count: number
  fertility: number
}

export type TokenizeResponse = Record<string, TokenizerResult>

export interface LeaderboardRow {
  tokenizer: string
  fertility: number
  vocabSize: number
  unkRate: number
  roundTripPass: boolean
}
