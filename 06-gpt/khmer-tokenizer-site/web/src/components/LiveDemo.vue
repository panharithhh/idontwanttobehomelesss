<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { fetchTokenizerNames, tokenize } from '../api'
import type { TokenizeResponse } from '../types'

const EXAMPLE_TEXT = 'អ្នកគ្រូបង្រៀនភាសាខ្មែរនៅសាលារៀន។'

const availableTokenizers = ref<string[]>([])
const selected = ref<Set<string>>(new Set())
const text = ref('')
const results = ref<TokenizeResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const listError = ref<string | null>(null)

onMounted(async () => {
  try {
    availableTokenizers.value = await fetchTokenizerNames()
    selected.value = new Set(availableTokenizers.value)
  } catch (err) {
    listError.value = err instanceof Error ? err.message : String(err)
  }
})

function toggle(name: string) {
  if (selected.value.has(name)) selected.value.delete(name)
  else selected.value.add(name)
}

function loadExample() {
  text.value = EXAMPLE_TEXT
}

async function run() {
  error.value = null
  results.value = null
  if (!text.value.trim() || selected.value.size === 0) return
  loading.value = true
  try {
    results.value = await tokenize(text.value, [...selected.value])
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section>
    <h2>Live demo</h2>
    <p v-if="listError" class="error-state">Could not reach the API: {{ listError }}</p>

    <textarea v-model="text" class="khmer-text" placeholder="Type or paste Khmer text here..." />

    <div class="tokenizer-checks">
      <label v-for="name in availableTokenizers" :key="name">
        <input type="checkbox" :checked="selected.has(name)" @change="toggle(name)" />
        {{ name }}
      </label>
    </div>

    <div class="actions">
      <button type="button" :disabled="loading || !text.trim() || selected.size === 0" @click="run">
        {{ loading ? 'Tokenizing…' : 'Tokenize' }}
      </button>
      <button class="secondary" type="button" @click="loadExample">Load example</button>
    </div>

    <p v-if="loading" class="loading-state">Running tokenizers…</p>
    <p v-if="error" class="error-state">{{ error }}</p>

    <div v-for="(result, name) in results" :key="name" class="demo-result">
      <div class="demo-result-header">
        <strong>{{ name }}</strong>
        <span>{{ result.count }} tokens · fertility {{ result.fertility.toFixed(2) }}</span>
      </div>
      <div class="khmer-text">
        <span v-for="(tok, i) in result.tokens" :key="i" class="token-chip">{{ tok }}</span>
      </div>
    </div>
  </section>
</template>
