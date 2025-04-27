import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:3000/api',
  timeout: 10000
})

export const fetchMasks = async (date: string) => {
  try {
    const response = await api.get(`/masks?date=${date}`)
    return response.data
  } catch (error) {
    console.error('Błąd pobierania maski:', error)
    throw error
  }
}

export const fetchAnalysis = async (startDate: string, endDate: string) => {
  try {
    const response = await api.get(`/analysis?start=${startDate}&end=${endDate}`)
    return response.data
  } catch (error) {
    console.error('Błąd pobierania analizy:', error)
    throw error
  }
}