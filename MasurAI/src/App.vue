<template>
  <div class="container">
    <!-- Niebieskie paski boczne -->
    <div class="side-decoration left"></div>
    <div class="side-decoration right"></div>

    <!-- Sekcja wyboru daty -->
    <div class="date-card">
      <h2>Wybierz zakres dat:</h2>
      <div class="date-wrapper">
        <div class="date-box">
          <label>Data początkowa:</label>
          <input 
            type="date" 
            v-model="dates[0]" 
            class="date-input"
            @change="fetchData"
          >
          <span class="formatted-date">{{ formatDate(dates[0]) }}</span>
        </div>
        <div class="date-box">
          <label>Data końcowa:</label>
          <input 
            type="date" 
            v-model="dates[1]" 
            class="date-input"
            @change="fetchData"
          >
          <span class="formatted-date">{{ formatDate(dates[1]) }}</span>
        </div>
      </div>
    </div>

    <!-- Sekcja z maską -->
    <div class="comparison-card">
      <h3>Maska satelitarna dla wybranego zakresu dat</h3>
      <div class="mask-container">
        <div class="mask-image">
          <img v-if="maskImage" :src="maskImage" alt="Generated mask">
          <img v-else :src="'/img/error-placeholder.png'" alt="Error placeholder">
        </div>
        <div class="date-badge">{{ formatDate(dates[0]) }} - {{ formatDate(dates[1]) }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { fetchMasks } from './services/api'

const dates = ref(['2020-01-05', '2022-02-28'])
const maskImage = ref<string | null>(null)

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
    const maskResult = await fetchMasks(dates.value[0], dates.value[1])
    maskImage.value = maskResult.imageUrl
  } catch (error) {
    console.error('Błąd pobierania danych:', error)
    maskImage.value = '/img/error-placeholder.png' // Ensure this path exists
  }
}

onMounted(() => {
  fetchData()
})

onUnmounted(() => {
  if (maskImage.value?.startsWith('blob:')) {
    URL.revokeObjectURL(maskImage.value)
  }
})
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  min-height: 100vh;
  font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
  background: linear-gradient(
    to right,
    #C7E4D9 1000px,
    #C7E4D9 100px,
    #C7E4D9 calc(100% - 450px),
    #C7E4D9 calc(100% - 450px)
  );
}


.side-decoration {
  position: fixed;
  top: 0;
  height: 100vh;
  width: 1000px;
  background: #D6EEEB;
  z-index: -1;
}

.side-decoration.left {
  left: 0;
}

.side-decoration.right {
  right: 0;
}

.date-card {
  background: #86d0c6;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
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

.date-box label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #2a348e;
}

.date-input {
  background-color: #D6EEEB;
  width: 200px;
  padding: 8px 12px;
  border: 2px solid #1294A7;
  border-radius: 8px;
  font-size: 14px;
}

.formatted-date {
  position: absolute;
  top: 50%;
  right: 15px;
  transform: translateY(-50%);
  color: #111111;
  pointer-events: none;
}

.comparison-card {
  background: #86d0c6;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.mask-container {
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  overflow: hidden;
  margin-top: 20px;
}

.mask-image {
  height: 400px;
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
  background: #86d0c6;
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
  color: #2a348e;
  margin: 0 0 16px 0;
}

h3 {
  color: #2a348e;
  margin: 0 0 20px 0;
}

.loading {
  color: #666;
  font-style: italic;
}
</style>