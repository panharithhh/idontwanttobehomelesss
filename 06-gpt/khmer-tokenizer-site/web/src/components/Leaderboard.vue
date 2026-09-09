<script setup lang="ts">
import { ref, computed } from 'vue'
import type { LeaderboardRow } from '../types'

type SortKey = keyof LeaderboardRow

const rows = ref<LeaderboardRow[]>([])
const sortKey = ref<SortKey>('fertility')
const sortAsc = ref(true)
const loadError = ref<string | null>(null)

// Static data only - no API call for the leaderboard, by design.
fetch(`${import.meta.env.BASE_URL}results.json`)
  .then((r) => {
    if (!r.ok) throw new Error(`failed to load results.json (${r.status})`)
    return r.json()
  })
  .then((data: LeaderboardRow[]) => {
    rows.value = data
  })
  .catch((err: Error) => {
    loadError.value = err.message
  })

const sortedRows = computed(() => {
  const copy = [...rows.value]
  copy.sort((a, b) => {
    const av = a[sortKey.value]
    const bv = b[sortKey.value]
    if (typeof av === 'string' && typeof bv === 'string') {
      return sortAsc.value ? av.localeCompare(bv) : bv.localeCompare(av)
    }
    const an = Number(av)
    const bn = Number(bv)
    return sortAsc.value ? an - bn : bn - an
  })
  return copy
})

function setSort(key: SortKey) {
  if (sortKey.value === key) {
    sortAsc.value = !sortAsc.value
  } else {
    sortKey.value = key
    sortAsc.value = true
  }
}

const columns: { key: SortKey; label: string }[] = [
  { key: 'tokenizer', label: 'Tokenizer' },
  { key: 'fertility', label: 'Fertility' },
  { key: 'vocabSize', label: 'Vocab size' },
  { key: 'unkRate', label: 'UNK rate' },
  { key: 'roundTripPass', label: 'Round-trip' },
]
</script>

<template>
  <section>
    <h2>Leaderboard</h2>
    <p v-if="loadError" class="error-state">{{ loadError }}</p>
    <div v-else class="card">
      <table>
        <thead>
          <tr>
            <th v-for="col in columns" :key="col.key" @click="setSort(col.key)">
              {{ col.label }}
              <span v-if="sortKey === col.key" class="arrow">{{ sortAsc ? '↑' : '↓' }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in sortedRows" :key="row.tokenizer">
            <td>{{ row.tokenizer }}</td>
            <td>{{ row.fertility.toFixed(2) }}</td>
            <td>{{ row.vocabSize.toLocaleString() }}</td>
            <td>{{ (row.unkRate * 100).toFixed(2) }}%</td>
            <td :class="row.roundTripPass ? 'pass' : 'fail'">
              {{ row.roundTripPass ? 'pass' : 'fail' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
