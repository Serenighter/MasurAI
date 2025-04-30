# <ins>MasurAI</ins>
<sub>Pictures taken using copernicus browser.</sub>
<sub>Images for the charts fetched from sentinelhub API.</sub>

### MasurAI is a set of python scripts that analyse Polish Masuria's lakes' water levels.
Image_analyze.py provides a Satellite Mask Comparison Tool for comparing satellite masks based on selected dates.
Api_analysis.py detects and tracks water level changes in lakes over time, specifically designed for Lake Tałty using Sentinel-2 API, then generates a chart based on the gathered data.

## Features

- Adaptive water detection using HSV color spaces
- Seasonal threshold adjustments based on image capture date
- Real-time visualization of water area trends
- Cloud interference reduction
- Contour-based shoreline detection
- Interactive display with dual-panel visualization
- Automatic parameter adjustment based on historical data
- Interpolated data
- Generate a chart image of changes in water levels

## Built using

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

## Example of the generated chart:
![Jezioro Tałty_water_levels](https://github.com/user-attachments/assets/3c724142-c67a-4dbf-82af-faae0c901de2)

## Example of mask comparison:
![2020-03-15-2024-03-09](https://github.com/user-attachments/assets/4613a08a-84cb-4bf8-9597-58780280f3da)

