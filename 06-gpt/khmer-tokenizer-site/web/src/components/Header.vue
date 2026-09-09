<script setup lang="ts">
// Update these three links before deploying.
const links = [
  { label: 'Paper', href: '#' },
  { label: 'Code', href: '#' },
  { label: 'Data', href: '#' },
]
</script>

<template>
  <section>
    <span class="kicker">NLP · Tokenization Benchmark</span>
    <h1>Khmer Tokenizer Fertility Benchmark</h1>
    <p>How efficiently do popular tokenizers segment Khmer text, and where do they break?</p>
    <nav>
      <a v-for="link in links" :key="link.label" class="button-link" :href="link.href">
        {{ link.label }}
      </a>
    </nav>
  </section>
</template>
