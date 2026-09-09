<script setup lang="ts">
interface TaxonomyExample {
  tag: string
  title: string
  description: string
}

// Content intentionally left as placeholders - fill these in yourself.
const examples: TaxonomyExample[] = [
  {
    tag: 'COENG',
    title: 'COENG split',
    description:
      'TODO: describe and show a worked example of a COENG (subscript consonant) sequence being split incorrectly.',
  },
  {
    tag: 'ZWSP',
    title: 'ZWSP failure',
    description:
      'TODO: describe and show a worked example of a zero-width space (word boundary) being mishandled.',
  },
  {
    tag: 'VOWEL',
    title: 'Vowel detachment',
    description:
      'TODO: describe and show a worked example of a dependent vowel sign being separated from its base consonant.',
  },
]
</script>

<template>
  <section>
    <h2>Error taxonomy</h2>
    <div v-for="ex in examples" :key="ex.title" class="taxonomy-example">
      <span class="kicker">{{ ex.tag }}</span>
      <h3>{{ ex.title }}</h3>
      <p class="todo-marker">{{ ex.description }}</p>
    </div>
  </section>
</template>
