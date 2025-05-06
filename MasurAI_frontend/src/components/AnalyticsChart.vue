<template>
  <div class="chart-container">
    <canvas ref="chartCanvas"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const props = defineProps<{
  chartData?: any
}>()

const chartCanvas = ref<HTMLCanvasElement | null>(null)
let chartInstance: Chart | null = null

watch(() => props.chartData, (newData) => {
  if (newData) updateChart(newData)
})

const updateChart = (data: any) => {
  if (chartInstance) chartInstance.destroy()
  
  if (chartCanvas.value) {
    chartInstance = new Chart(chartCanvas.value, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [{
          label: 'Parametry wody',
          data: data.values,
          borderColor: '#2196F3',
          tension: 0.4
        }]
      }
    })
  }
}
</script>

<style>
.chart-container {
  background: white;
  padding: 20px;
  border-radius: 15px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>