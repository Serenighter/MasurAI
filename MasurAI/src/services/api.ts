import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8080/api',
  timeout: 10000
})

export const fetchMasks = async (startDate: string, endDate: string) => {
  try {
    const response = await api.get(`/image/date/${formatDate(startDate)}/${formatDate(endDate)}`, {
      responseType: 'blob'
    })
    
    const imageUrl = URL.createObjectURL(response.data)
    return { imageUrl, startDate, endDate }
  } catch (error) {
    console.error('Błąd pobierania maski:', error)
    throw error
  }
}

export const fetchAnalysis = async (start: string, end: string) => {
  try {
    const response = await api.get('/analysis', {
      params: {
        start: formatDate(start),
        end: formatDate(end)
      }
    })
    return response.data
  } catch (error) {
    console.error('Błąd analizy:', error)
    throw error
  }
}

const formatDate = (dateString: string) => {
  const date = new Date(dateString)
  return [
    date.getFullYear(),
    (date.getMonth() + 1).toString().padStart(2, '0'),
    date.getDate().toString().padStart(2, '0')
  ].join('-')
}