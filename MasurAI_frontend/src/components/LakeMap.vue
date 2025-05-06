<template>
  <div class="map-container">
    <div class="map" ref="mapContainer"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const props = defineProps<{
  coordinates: [number, number]
}>()

const mapContainer = ref<HTMLElement | null>(null)

onMounted(() => {
  if (mapContainer.value) {
    const map = L.map(mapContainer.value).setView(props.coordinates, 12)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map)
  }
})
</script>

<style>
.map-container {
  height: 100%;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.map {
  height: 100%;
  width: 100%;
}
</style>