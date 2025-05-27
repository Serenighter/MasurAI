<template>
  <div class="container">
    <div class="side-decoration left"></div>
    <div class="side-decoration right"></div>

    <div class="date-card">
      <h2>Wybierz zakres dat:</h2>
      
      <div class="calendar-wrapper">
        <VCalendar
          :min-date="new Date(2020, 0, 1)"
          :max-date="new Date()"
          :attributes="calendarAttributes"
          first-day-of-week="1"
          @dayclick="handleDateClick"
          is-expanded
        >
          <template #day-content="{ day }">
            <div class="day-content">
              {{ day.day }}
            </div>
          </template>
        </VCalendar>
      </div>
      <div class="date-wrapper">
        <div class="date-box">
          <label>Data początkowa:</label>
          <DatePicker
            v-model="dates[0]"
            :enable-time-picker="false"
            :allowed-dates="availableDates"
            :highlight="highlightConfig"
            @update:model-value="fetchData"
            locale="pl"
            class="custom-datepicker"
          />
        </div>
        <div class="date-box">
          <label>Data końcowa:</label>
          <DatePicker
            v-model="dates[1]"
            :enable-time-picker="false"
            :allowed-dates="availableDates"
            :highlight="highlightConfig"
            @update:model-value="fetchData"
            locale="pl"
            class="custom-datepicker"
          />
        </div>
        </div>
      </div>

    <div class="comparison-card">
      <h3>Maski dla wybranych dat</h3>
      <div class="mask-container">
        <div class="mask-image">
          <div v-if="loading" class="loading-anim">
            <div class="spinner"></div>
            <span class="loading-text">Ładowanie obrazu...</span>
          </div>
          <template v-else>
            <img 
              v-if="maskImage"
              :src="maskImage" 
              alt="Generated mask"
              class="fade-in"
            >
            <div v-else class="error-message">
              <span>  Brak dostępnego obrazu</span>
              <p>Wybierz inne daty dla porównania masek</p>
            </div>
          </template>
        </div>
        <div class="date-badge">{{ formatDate(dates[0]) }} & {{ formatDate(dates[1]) }}</div>
      </div>
    </div>
  <div class="chart-card">
      <button 
          @click="toggleStaticImage"
          class="chart-button"
        >
          {{ showStaticImage ? 'Ukryj wykres' : 'Pokaż wykres' }}
      </button>

      <div v-if="showStaticImage" class="static-image-container fade-in-animation">
        <img 
          src="/img/Talty_predictions.png" 
          alt="Wykres"
          class="static-image"
        >
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import VCalendar from 'v-calendar'
import { fetchMasks } from './services/api'
import DatePicker from '@vuepic/vue-datepicker'
import 'v-calendar/style.css'
import '@vuepic/vue-datepicker/dist/main.css'
import { AVAILABLE_DATES } from '@/config/dates';

const availableDatesRaw = AVAILABLE_DATES
const showStaticImage = ref(false)
const loading = ref(false)
const dates = ref<string[]>([availableDatesRaw[4], availableDatesRaw[15]])
const maskImage = ref<string | null>(null)

const highlightConfig = computed(() => ({
  dates: availableDates.value,
  color: '#86d0c6',
  style: {
    backgroundColor: '#86d0c6',
    borderRadius: '4px'
  }
}))
const availableDates = computed(() => 
  availableDatesRaw.map(date => new Date(date))
)

const formatDate = (dateString: string) => 
  new Date(dateString).toLocaleDateString('pl-PL', { 
    day: '2-digit', 
    month: '2-digit', 
    year: 'numeric' 
  })

const formatDateToYYYYMMDD = (date: Date) => 
  date.toISOString().split('T')[0]

const calendarAttributes = computed(() => {
const startDate = new Date(dates.value[0])
const endDate = new Date(dates.value[1])
  
  return [
    {
      key: 'available',
      dot: { color: 'green', class: 'available-date' },
      dates: availableDates.value
    },
    {
      key: 'selected',
      highlight: { 
        color: 'blue', 
        fillMode: 'solid', 
        class: 'selected-date' 
      },
      dates: startDate.getTime() === endDate.getTime() 
        ? [startDate] 
        : [{ start: startDate, end: endDate }]
    }
  ]
})

const fetchData = async () => {
  try {
    loading.value = true
    const maskResult = await fetchMasks(dates.value[0], dates.value[1])
    maskImage.value = maskResult.imageUrl
  } catch (error) {
    console.error('Błąd pobierania danych:', error)
    maskImage.value = null
  } finally {
    loading.value = false
  }
}

const handleDateClick = (day: { date: Date }) => {
  const clickedDate = formatDateToYYYYMMDD(day.date)
  if (availableDatesRaw.includes(clickedDate)) {
    dates.value = [clickedDate, clickedDate]
    fetchData()
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
const toggleStaticImage = () => {
  showStaticImage.value = !showStaticImage.value
}
</script>

<style scoped>
.container {
  max-width: 1600px;
  margin: 0 auto;
  padding: 50px;
  min-height: 105vh;
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
  margin-top: 30px;
}

.mask-image {
  height: 70vh;
  min-height: 400px;
  background: #f1f3f5;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.mask-image img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
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

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  z-index: 10;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #2a348e;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto;
  align-self: center;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  color: #2a348e;
  font-weight: 500;
  text-align: center;
}

.fade-in {
  animation: fadeIn 0.7s ease-in;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.chart-card {
  background: #86d0c6;
  border-radius: 12px;
  padding: 24px;
  margin-top: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  margin-bottom: 10vh;
}

.fade-in-animation {
  animation: fadeIn 0.7s ease-in;
}

.static-image-container {
  margin-top: 20px;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  overflow: hidden;
}

.static-image {
  width: 100%;
  height: auto;
  display: block;
}

.chart-button {
  display: block;
  margin: 0 auto;
  padding: 10px 20px;
  background: #2a348e;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}

.chart-button:hover {
  background: #1a236e;
}
</style>