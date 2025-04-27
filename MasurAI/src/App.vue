<template>
  <div class="container">
    <!-- Niebieskie paski boczne -->
    <div class="side-decoration left"></div>
    <div class="side-decoration right"></div>

    <!-- Sekcja wyboru daty -->
    <div class="date-card">
      <h2>Wybierz datę:</h2>
      <div class="date-wrapper">
        <div class="date-box" v-for="(date, index) in dates" :key="index">
          <input 
            type="date" 
            v-model="dates[index]" 
            class="date-input"
            @change="fetchData"
          >
          <span class="formatted-date">{{ formatDate(date) }}</span>
        </div>
      </div>
    </div>

    <!-- Sekcja porównania masek -->
    <div class="comparison-card">
      <h3>Porównanie dwóch masek</h3>
      <div class="mask-grid">
        <div 
          v-for="(mask, index) in masks" 
          :key="index" 
          class="mask-card"
        >
          <div class="mask-image">
            <img v-if="mask" :src="mask" alt="Maska satelitarna">
            <div v-else class="loading">Ładowanie obrazu...</div>
          </div>
          <div class="date-badge">{{ formatDate(dates[index]) }}</div>
        </div>
      </div>
    </div>

    <!-- Sekcja wyników -->
    <div class="results-card">
      <h3>Wynik:</h3>
      <ul class="results-list">
        <li 
          v-for="(result, index) in analysisResults" 
          :key="index" 
          class="result-item"
        >
          <div class="result-header">
            <span class="location">{{ result.location }}</span>
            <span class="date">{{ result.date }}</span>
          </div>
          <div class="result-content">
            <span class="parameter">{{ result.parameter }}</span>
            <span 
              class="status"
              :class="{'interpolated': result.type === 'interpolated'}"
            >
              {{ result.type === 'interpolated' ? 'Interpolowane' : 'Obserwowane' }}
            </span>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { fetchMasks, fetchAnalysis } from './services/api'

const dates = ref(['2021-10-28', '2025-01-28'])
const masks = ref<string[]>([])
const analysisResults = ref<any[]>([])

const formatDate = (dateString: string) => {
  const options: Intl.DateTimeFormatOptions = { 
    day: '2-digit', 
    month: '2-digit', 
    year: 'numeric' 
  }
  return new Date(dateString).toLocaleDateString('pl-PL', options)
}

const fetchData = async () => {
  try {
    const maskPromises = dates.value.map(date => fetchMasks(date))
    const maskResults = await Promise.all(maskPromises)
    masks.value = maskResults.map(res => res.imageUrl)

    const analysisData = await fetchAnalysis(dates.value[0], dates.value[1])
    analysisResults.value = analysisData.results
  } catch (error) {
    console.error('Błąd pobierania danych:', error)
  }
}
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  min-height: 100vh;
  background: linear-gradient(
    to right,
    #e3f2fd 250px,
    #f8f9fa 250px,
    #f8f9fa calc(100% - 250px),
    #e3f2fd calc(100% - 250px)
  );
}

.side-decoration {
  position: fixed;
  top: 0;
  height: 100vh;
  width: 250px;
  background: #e3f2fd;
  z-index: -1;
}

.side-decoration.left {
  left: 0;
}

.side-decoration.right {
  right: 0;
}

.date-card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.date-wrapper {
  display: flex;
  gap: 20px;
  margin-top: 16px;
}

.date-box {
  flex: 1;
  position: relative;
}

.date-input {
  width: 200px;
  padding: 8px 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 14px;
}

.formatted-date {
  position: absolute;
  top: 50%;
  right: 15px;
  transform: translateY(-50%);
  color: #666;
  pointer-events: none;
}

.comparison-card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.mask-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
  margin-top: 20px;
}

.mask-card {
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  overflow: hidden;
}

.mask-image {
  height: 300px;
  background: #f1f3f5;
  display: flex;
  align-items: center;
  justify-content: center;
}

.mask-image img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.date-badge {
  padding: 8px 16px;
  background: rgba(0,0,0,0.8);
  color: white;
  display: inline-block;
  border-radius: 0 0 8px 8px;
  margin: -5px 0 0 0;
}

.results-card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.results-list {
  list-style: none;
  padding: 0;
  margin: 20px 0 0;
}

.result-item {
  padding: 16px;
  border-bottom: 1px solid #eee;
}

.result-item:last-child {
  border-bottom: none;
}

.result-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.location {
  font-weight: 500;
  color: #2c3e50;
}

.date {
  color: #666;
  font-size: 0.9em;
}

.result-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.parameter {
  color: #4a5568;
}

.status {
  font-weight: 500;
  padding: 4px 12px;
  border-radius: 20px;
  background: #e3f2fd;
  color: #1976d2;
}

.status.interpolated {
  background: #e8f5e9;
  color: #2e7d32;
}

h2 {
  color: #1a237e;
  margin: 0 0 16px 0;
}

h3 {
  color: #2c3e50;
  margin: 0 0 20px 0;
}

.loading {
  color: #666;
  font-style: italic;
}
</style>