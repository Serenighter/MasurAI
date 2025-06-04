# <ins>MasurAI</ins>
# http://masurai.sytes.net:3000/
<sub>Pictures taken using copernicus browser.</sub>
<sub>Images for the charts fetched from sentinelhub API.</sub>

## Comprehensive water level analysis system for Polish Masuria's lakes with web interface
### MasurAI is a set of python scripts that analyse Polish Masuria's lakes' water levels.
Image_analyze.py provides a Satellite Mask Comparison Tool for comparing satellite masks based on selected dates.
Api_analysis.py detects and tracks water level changes in lakes over time, specifically designed for Lake Tałty using Sentinel-2 API, then generates a chart based on the gathered data.
Water_level_predict.py implements time series forecasting using Facebook's Prophet algorithm to predict future water levels based on historical data patterns.

### Python Analysis Core
- `image_analyze.py`: Satellite Mask Comparison Tool
- `api_analysis.py`: Water level detection and chart generation
- `water_level_predict.py`: Time series forecasting and prediction analysis
- Features:
  - Adaptive HSV color space detection
  - Seasonal threshold adjustments
  - Real-time visualization of water area trends
  - Cloud interference reduction
  - Contour-based shoreline detection
  - Interactive display with dual-panel visualization
  - Chart generation of changes in water levels
  - Prophet-based forecasting with seasonal patterns
  - Statistical analysis and trend prediction

### Web Application Interface
Built using **Java Spring Boot Backend** (REST API) and **Vue.js 3 Frontend**

## Built using

### Backend
- Java 17
- Spring Boot 3
- Maven

### Frontend
- Vue.js 3
- Pinia
- Vue Router
- Vite Build System
- [V Calendar](https://vcalendar.io/)
- [Vue Datepicker](https://vue3datepicker.com/)

### Python Analysis Core
- Python 3.13

- OpenCV (`opencv-python`)
- Matplotlib (`matplotlib`)
- pandas (`pandas`)
- scikit-image (`scikit-image`)
- tqdm (`tqdm`)
- NumPy (`numpy`)
- rasterio (`rasterio`)
- geopandas (`geopandas`)
- shapely (`shapely`)
- dateutil (`python-dateutil`)
- requests (`requests`)
- oauthlib (`oauthlib`)
- scipy (`scipy`)
- Python 3.11
- prophet (`prophet`)
- seaborn (`seaborn`)

## Example of the generated chart:
![Talty_predictions](https://github.com/user-attachments/assets/2d0cf4a4-b756-4ce3-a36e-2a1f02493378)

## Example of mask comparison:
![2020-03-15-2024-03-09](https://github.com/user-attachments/assets/4613a08a-84cb-4bf8-9597-58780280f3da)

