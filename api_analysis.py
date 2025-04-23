import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import Polygon
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from scipy.interpolate import interp1d
from dateutil.relativedelta import relativedelta
from matplotlib.gridspec import GridSpec
import requests
import json
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

class LakeMonitor:
    def __init__(self, client_id, client_secret, lake_name, coordinates):
        """
        Initialize the Lake Monitor with credentials and lake information.
        Args:
            client_id (str): Sentinel Hub OAuth client ID
            client_secret (str): Sentinel Hub OAuth client secret
            lake_name (str): Name of the lake to monitor
            coordinates (list): List of (lon, lat) tuples defining lake boundary
        """
        self.lake_name = lake_name
        self.coordinates = coordinates
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.lake_geometry = Polygon(coordinates)
        self.results_dir = f'{lake_name}_results'
        self.images_dir = f'{self.results_dir}/images'
        self.data_dir = 'data'
        
        # Create necessary directories
        for directory in [self.results_dir, self.images_dir, self.data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.authenticate()
            
    def authenticate(self):
        #Sentinel Hub OAuth2 service, get access token.
        try:
            # Create OAuth2 client
            client = BackendApplicationClient(client_id=self.client_id)
            oauth = OAuth2Session(client=client)
            
            # Get token
            token_url = 'https://services.sentinel-hub.com/oauth/token'
            self.token = oauth.fetch_token(
                token_url=token_url,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            
            print("Successfully authenticated with Sentinel Hub API")
            return self.token
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return None
    
    def create_lake_geojson(self):
        #Create a GeoJSON file for the lake.
        lake_gdf = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', 
                                    geometry=[self.lake_geometry])
        geojson_path = f'{self.results_dir}/{self.lake_name}_boundary.geojson'
        lake_gdf.to_file(geojson_path, driver='GeoJSON')
        return geojson_path
    
    def fetch_satellite_data(self, start_date, end_date, cloud_percent=30):
        """
        Fetch Sentinel-2 data for the specified date range using Sentinel Hub API.
        Args:
            start_date (str): Start date in format 'YYYYMMDD'
            end_date (str): End date in format 'YYYYMMDD'
            cloud_percent (int): Maximum cloud percentage to accept
            
        Returns:
            dict: List of available products
        """
        print(f"Fetching Sentinel-2 data for {self.lake_name} from {start_date} to {end_date}...")
        
        # Format dates for Sentinel Hub API (YYYY-MM-DD)
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # Convert lake geometry to GeoJSON format
        geometry = self.lake_geometry.__geo_interface__
        
        # Define the request body
        catalog_request = {
            "collections": ["sentinel-2-l2a"],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "intersects": geometry,
            "limit": 100,  # Maximum number of results
            "query": {
                "eo:cloud_cover": {
                    "lte": cloud_percent
                }
            }
        }
        
        # Make request to Sentinel Hub Catalog API
        try:
            if not self.token:
                self.authenticate()
                
            headers = {
                'Authorization': f"Bearer {self.token['access_token']}",
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                "https://services.sentinel-hub.com/api/v1/catalog/search",
                headers=headers,
                json=catalog_request
            )
            
            if response.status_code == 200:
                products = response.json()
                print(f"Found {len(products.get('features', []))} Sentinel-2 scenes.")
                return products
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
                return {"features": []}
                
        except Exception as e:
            print(f"Error in fetch_satellite_data: {str(e)}")
            return {"features": []}
    
    def download_product(self, feature):
        """
        Download a Sentinel-2 product using Process API.
        Args:
            feature (dict): Feature from catalog search results    
        Returns:
            str: Path to downloaded data
        """
        try:
            if not self.token:
                self.authenticate()
                
            # Extract date and ID from feature
            properties = feature.get('properties', {})
            date_str = properties.get('datetime', '').split('T')[0]
            product_id = feature.get('id', 'unknown')
            
            # Create a directory for this product
            product_dir = f"{self.data_dir}/{product_id}"
            if not os.path.exists(product_dir):
                os.makedirs(product_dir)
                
            # Define bands to download (B02=Blue, B03=Green, B04=Red, B08=NIR, SCL=Scene Classification)
            bands = ["B02", "B03", "B04", "B08", "SCL"]
            
            # Extract geometry from feature
            bbox = None
            if 'bbox' in feature:
                # Use bbox if available
                bbox = feature['bbox']
            elif 'geometry' in feature:
                # Calculate bbox from geometry
                geometry = feature['geometry']
                # simplified - in real application calculate proper bbox
                coordinates = geometry.get('coordinates', [[[]]])[0][0]
                if coordinates:
                    lons = [coord[0] for coord in coordinates]
                    lats = [coord[1] for coord in coordinates]
                    bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            if not bbox:
                # Use lake geometry if no bbox is available
                geom = self.lake_geometry.bounds
                bbox = [geom[0], geom[1], geom[2], geom[3]]
            
            # Download each band
            band_paths = {}
            for band in bands:
                # Define request data for specific band
                process_request = {
                    "input": {
                        "bounds": {
                            "bbox": bbox,
                            "properties": {
                                "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                            }
                        },
                        "data": [{
                            "dataFilter": {
                                "timeRange": {
                                    "from": f"{date_str}T00:00:00Z",
                                    "to": f"{date_str}T23:59:59Z"
                                }
                            },
                            "type": "sentinel-2-l2a"
                        }]
                    },
                    "output": {
                        "width": 1000,
                        "height": 1000
                    },
                    "evalscript": f"""
                    //VERSION=3
                    function setup() {{
                        return {{
                            input: ["{band}"],
                            output: {{ bands: 1 }}
                        }};
                    }}

                    function evaluatePixel(sample) {{
                        return [sample.{band}];
                    }}
                    """
                }
                
                # Make request to Process API
                headers = {
                    'Authorization': f"Bearer {self.token['access_token']}",
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(
                    "https://services.sentinel-hub.com/api/v1/process",
                    headers=headers,
                    json=process_request
                )
                
                if response.status_code == 200:
                    # Save band as GeoTIFF
                    band_path = f"{product_dir}/{band}.tif"
                    with open(band_path, 'wb') as f:
                        f.write(response.content)
                    band_paths[band] = band_path
                    print(f"Downloaded {band} for {date_str}")
                else:
                    print(f"Error downloading {band}: {response.status_code} - {response.text}")
            
            return product_dir, band_paths, date_str
                
        except Exception as e:
            print(f"Error in download_product: {str(e)}")
            return None, {}, ""
    
    def download_products(self, products, limit=None):
        """
        Download Sentinel-2 products.
        Args:
            products (dict): Dictionary of products from fetch_satellite_data
            limit (int, optional): Limit the number of products to download
        Returns:
            list: List of downloaded product paths and metadata
        """
        # Extract features
        features = products.get('features', [])
        
        # Sort by cloud cover
        features.sort(key=lambda x: x.get('properties', {}).get('eo:cloud_cover', 100))
        
        # Limit the number if needed
        if limit is not None and limit < len(features):
            features = features[:limit]
        
        # Download products
        product_data = []
        for feature in features:
            print(f"Downloading product {feature.get('id', 'unknown')}...")
            product_dir, band_paths, date_str = self.download_product(feature)
            
            if product_dir and band_paths:
                cloud_cover = feature.get('properties', {}).get('eo:cloud_cover', None)
                product_data.append({
                    'path': product_dir,
                    'band_paths': band_paths,
                    'date': date_str,
                    'cloud_cover': cloud_cover,
                    'feature': feature
                })
        
        return product_data
    
    def calculate_ndwi(self, green_band, nir_band):
        """
        Calculate NDWI (Normalized Difference Water Index).
        Args:
            green_band (numpy.array): Green band (B03 for Sentinel-2)
            nir_band (numpy.array): NIR band (B08 for Sentinel-2)
        Returns:
            numpy.array: NDWI values
        """
        # Avoid division by zero
        mask = (green_band + nir_band) != 0
        ndwi = np.zeros_like(green_band, dtype=np.float32)
        ndwi[mask] = (green_band[mask] - nir_band[mask]) / (green_band[mask] + nir_band[mask])
        return ndwi
    
    def create_rgb_preview(self, band_paths, date_str, save=True):
        """
        Create an RGB preview image from Sentinel-2 bands.
        Args:
            band_paths (dict): Dictionary of band file paths
            date_str (str): Date string
            save (bool): Whether to save the image
        Returns:
            tuple: (RGB array, image path if saved)
        """
        # Read the bands
        with rasterio.open(band_paths['B04']) as src:
            red_band = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            
        with rasterio.open(band_paths['B03']) as src:
            green_band = src.read(1).astype(np.float32)
            
        with rasterio.open(band_paths['B02']) as src:
            blue_band = src.read(1).astype(np.float32)
        
        # Create RGB composite
        # Normalize and stack bands
        def normalize(band):
            min_val = np.percentile(band, 2)
            max_val = np.percentile(band, 98)
            return np.clip((band - min_val) / (max_val - min_val), 0, 1)
        
        rgb = np.dstack((normalize(red_band), 
                         normalize(green_band), 
                         normalize(blue_band)))
        
        # Format date
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        if save:
            # Save the image
            img_path = f"{self.images_dir}/{self.lake_name}_{date}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb)
            plt.title(f"{self.lake_name} - {date}")
            plt.axis('off')
            plt.savefig(img_path, dpi=200, bbox_inches='tight')
            plt.close()
            return rgb, img_path
        
        return rgb, None
    
    def create_ndwi_preview(self, band_paths, date_str, save=True):
        """
        Create an NDWI preview image from Sentinel-2 bands.
        Args:
            band_paths (dict): Dictionary of band file paths
            date_str (str): Date string
            save (bool): Whether to save the image
        Returns:
            tuple: (NDWI array, image path if saved)
        """
        # Read the bands
        with rasterio.open(band_paths['B03']) as src:
            green_band = src.read(1).astype(np.float32)
            
        with rasterio.open(band_paths['B08']) as src:
            nir_band = src.read(1).astype(np.float32)
        
        # Calculate NDWI
        ndwi = self.calculate_ndwi(green_band, nir_band)
        
        # Format date
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        if save:
            # Save the image
            img_path = f"{self.images_dir}/{self.lake_name}_NDWI_{date}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(ndwi, cmap='RdYlBu')
            plt.colorbar(label='NDWI')
            plt.title(f"{self.lake_name} NDWI - {date}")
            plt.axis('off')
            plt.savefig(img_path, dpi=200, bbox_inches='tight')
            plt.close()
            return ndwi, img_path
        
        return ndwi, None
    
    def analyze_water_level(self, product_info, save_previews=True):
        """
        Analyze water level from a Sentinel-2 product using NDWI.
        Args:
            product_info (dict): Product information with path and band paths
            save_previews (bool): Whether to save preview images
        Returns:
            tuple: (date, water_area_km2, water_pixels, cloud_percentage, preview_paths)
        """
        band_paths = product_info['band_paths']
        date_str = product_info['date']
        
        # Read the bands
        with rasterio.open(band_paths['B03']) as src:
            green_band = src.read(1)
            transform = src.transform
            crs = src.crs
            
        with rasterio.open(band_paths['B08']) as src:
            nir_band = src.read(1)
            
        with rasterio.open(band_paths['SCL']) as src:
            scl = src.read(1)
            
        # Calculate NDWI
        ndwi = self.calculate_ndwi(green_band, nir_band)
        
        # Apply cloud mask using Scene Classification Layer
        # SCL values 1,2,3,8,9,10 - various types of clouds/shadows
        cloud_mask = np.isin(scl, [1, 2, 3, 8, 9, 10])
        
        # Water is typically NDWI > 0, but threshold can be adjusted
        water_mask = ndwi > 0
        
        # Mask out clouds
        valid_pixels = ~cloud_mask
        water_mask = water_mask & valid_pixels
        
        # Calculate water area
        pixel_area_m2 = abs(transform[0] * transform[4])  # Area of one pixel in square meters
        water_pixels = np.sum(water_mask)
        water_area_m2 = water_pixels * pixel_area_m2
        water_area_km2 = water_area_m2 / 1_000_000  # Convert to square kilometers
        
        # Calculate cloud percentage
        cloud_percentage = (np.sum(cloud_mask) / cloud_mask.size) * 100
        
        # Convert date string to datetime
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        
        # Create preview images
        preview_paths = {}
        if save_previews:
            # RGB preview
            _, rgb_path = self.create_rgb_preview(band_paths, date_str)
            preview_paths['rgb'] = rgb_path
            
            # NDWI preview
            _, ndwi_path = self.create_ndwi_preview(band_paths, date_str)
            preview_paths['ndwi'] = ndwi_path
            
            # Water mask preview
            water_mask_path = f"{self.images_dir}/{self.lake_name}_water_mask_{date_str}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(water_mask, cmap='Blues')
            plt.title(f"{self.lake_name} Water Mask - {date_str}")
            plt.axis('off')
            plt.savefig(water_mask_path, dpi=200, bbox_inches='tight')
            plt.close()
            preview_paths['water_mask'] = water_mask_path
        
        return date, water_area_km2, water_pixels, cloud_percentage, preview_paths
    
    def analyze_all_products(self, product_data, save_previews=True):
        """
        Analyze water levels for all downloaded products.
        Args:
            product_data (list): List of product information dictionaries
            save_previews (bool): Whether to save preview images
        Returns:
            pd.DataFrame: DataFrame with water level data
        """
        results = []
        
        for product_info in product_data:
            try:
                date, water_area, water_pixels, cloud_percentage, preview_paths = self.analyze_water_level(
                    product_info, save_previews
                )
                results.append({
                    'date': date,
                    'water_area_km2': water_area,
                    'water_pixels': water_pixels,
                    'cloud_percentage': cloud_percentage,
                    'rgb_preview': preview_paths.get('rgb', ''),
                    'ndwi_preview': preview_paths.get('ndwi', ''),
                    'water_mask_preview': preview_paths.get('water_mask', '')
                })
                print(f"Analyzed {date}: Water area = {water_area:.2f} km²")
            except Exception as e:
                print(f"Error analyzing {product_info['path']}: {str(e)}")
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(results)
        if not df.empty:
            df.sort_values('date', inplace=True)
            
            # Save results to CSV
            df.to_csv(f'{self.results_dir}/{self.lake_name}_water_levels.csv', index=False)
        
        return df
    
    def interpolate_missing_data(self, df, frequency='1M'):

        if df.empty or len(df) < 2:
            return pd.DataFrame()
        
        # Handle any duplicate dates by taking the mean of measurements from the same day
        df = df.groupby('date').agg({
            'water_area_km2': 'mean', 
            'water_pixels': 'mean',
            'cloud_percentage': 'mean'
        }).reset_index()
    
        # Set date as index
        df = df.set_index('date')
    
        # Create a date range with the desired frequency
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=frequency)
    
        # Reindex
        df_reindexed = df.reindex(date_range)
    
        # Choose interpolation method based on available data points
        interp_method = 'linear'  # Default to linear which needs only 2 points
        if len(df) >= 4:  # Only use cubic if we have enough data points
            try:
                # Try cubic first
                df_reindexed['water_area_km2'] = df_reindexed['water_area_km2'].interpolate(method='cubic')
            except ValueError:
                # Fall back to linear if cubic fails
                print("Warning: Cubic interpolation failed, using linear interpolation instead.")
                df_reindexed['water_area_km2'] = df_reindexed['water_area_km2'].interpolate(method='linear')
        else:
            # Use linear interpolation for fewer data points
            df_reindexed['water_area_km2'] = df_reindexed['water_area_km2'].interpolate(method='linear')
    
        # For other columns, always use linear interpolation
        for col in df_reindexed.columns:
            if col != 'water_area_km2' and col in df.columns:
                df_reindexed[col] = df_reindexed[col].interpolate(method='linear')
        
        # Reset index to make date a column again
        df_interpolated = df_reindexed.reset_index()
        df_interpolated = df_interpolated.rename(columns={'index': 'date'})
        
        return df_interpolated
    
    def filter_cloud_coverage(self, df, max_cloud_percent=50):
        """
        Filter out observations with high cloud coverage.
        Args:
            df (pd.DataFrame): DataFrame with water level data
            max_cloud_percent (float): Maximum cloud percentage to keep
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if df.empty:
            return df
        return df[df['cloud_percentage'] <= max_cloud_percent]
    
    def plot_water_levels(self, df, interpolated_df=None, title=None):
        """
        Plot water level changes over time.
        Args:
            df (pd.DataFrame): DataFrame with water level data
            interpolated_df (pd.DataFrame, optional): DataFrame with interpolated data
            title (str, optional): Title for the plot
        Returns:
            str: Path to saved plot
        """
        if df.empty:
            print("No data to plot")
            return None
            
        plt.figure(figsize=(12, 6))
        
        # Plot actual data points
        plt.scatter(df['date'], df['water_area_km2'], color='blue', label='Observed')
        
        # Plot original line
        plt.plot(df['date'], df['water_area_km2'], 'b-', alpha=0.5)
        
        # Plot interpolated data if provided
        if interpolated_df is not None and not interpolated_df.empty:
            plt.plot(interpolated_df['date'], interpolated_df['water_area_km2'], 
                    'r--', label='Interpolated')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Water Area (km²)')
        if title:
            plt.title(title)
        else:
            plt.title(f'{self.lake_name} Water Level Changes')
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plot_path = f'{self.results_dir}/{self.lake_name}_water_levels.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Save as SVG for web embedding
        svg_path = f'{self.results_dir}/{self.lake_name}_water_levels.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.show()
        plt.close()
        
        return plot_path
    
    def create_combined_visualization(self, df, n_samples=4):
        """
        Create a combined visualization with water level chart and sample images.
        Args:
            df (pd.DataFrame): DataFrame with water level data
            n_samples (int): Number of sample images to include
        Returns:
            str: Path to saved plot
        """
        if df.empty or len(df) < 1:
            print("Not enough data for combined visualization")
            return None
            
        # Select n_samples evenly spaced samples
        if len(df) <= n_samples:
            samples = df
        else:
            indices = np.linspace(0, len(df) - 1, n_samples, dtype=int)
            samples = df.iloc[indices]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, n_samples + 1, figure=fig, height_ratios=[2, 1])
        
        # Plot water level chart in top row
        ax_chart = fig.add_subplot(gs[0, :])
        ax_chart.scatter(df['date'], df['water_area_km2'], color='blue', label='Observed')
        ax_chart.plot(df['date'], df['water_area_km2'], 'b-', alpha=0.5)
        
        # Mark the sample points on the chart
        ax_chart.scatter(samples['date'], samples['water_area_km2'], color='red', s=100, 
                        zorder=5, label='Samples shown below')
        
        ax_chart.set_xlabel('Date')
        ax_chart.set_ylabel('Water Area (km²)')
        ax_chart.set_title(f'{self.lake_name} Water Level Changes')
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_chart.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()
        ax_chart.legend()
        ax_chart.grid(True, alpha=0.3)
        
        # Plot sample images in bottom row
        for i, (_, sample) in enumerate(samples.iterrows()):
            if i < n_samples:  # Ensure we don't exceed the grid
                # Check if the preview image exists
                if os.path.exists(sample['rgb_preview']):
                    ax_img = fig.add_subplot(gs[1, i])
                    img = plt.imread(sample['rgb_preview'])
                    ax_img.imshow(img)
                    ax_img.set_title(sample['date'].strftime('%Y-%m-%d'))
                    ax_img.axis('off')
        
        # Add NDWI legend to the last position
        if len(samples) > 0 and os.path.exists(samples.iloc[0]['ndwi_preview']):
            ax_ndwi = fig.add_subplot(gs[1, -1])
            img = plt.imread(samples.iloc[0]['ndwi_preview'])
            ax_ndwi.imshow(img)
            ax_ndwi.set_title('NDWI Example')
            ax_ndwi.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        combined_path = f'{self.results_dir}/{self.lake_name}_combined_visualization.png'
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        
        # Save as SVG for web embedding
        combined_svg_path = f'{self.results_dir}/{self.lake_name}_combined_visualization.svg'
        plt.savefig(combined_svg_path, format='svg', bbox_inches='tight')
        plt.show()
        plt.close()
        
        return combined_path
    
    def run_monitoring(self, start_date, end_date, cloud_percent=30, limit=None, save_previews=True):
        """
        Run the full monitoring pipeline.
        Args:
            start_date (str): Start date in format 'YYYYMMDD'
            end_date (str): End date in format 'YYYYMMDD'
            cloud_percent (int): Maximum cloud percentage for initial query
            limit (int, optional): Limit the number of products to download
            save_previews (bool): Whether to save preview images
        Returns:
            pd.DataFrame: DataFrame with water level data
        """
        # Create lake GeoJSON
        self.create_lake_geojson()
        
        # Fetch satellite data
        products = self.fetch_satellite_data(start_date, end_date, cloud_percent)
        
        # Download products
        product_data = self.download_products(products, limit)
        
        # Analyze water levels
        df = self.analyze_all_products(product_data, save_previews)
        
        if df.empty:
            print("No data could be analyzed. Check authentication and area of interest.")
            return df, pd.DataFrame()
        
        # Filter by cloud coverage
        df_filtered = self.filter_cloud_coverage(df)
        
        # Interpolate missing data
        df_interpolated = self.interpolate_missing_data(df_filtered)
        begin_date = f"{start_date[6:8]}.{start_date[4:6]}.{start_date[0:4]}"
        ending_date = f"{end_date[6:8]}.{end_date[4:6]}.{end_date[0:4]}"
        # Plot water levels
        plot_path = self.plot_water_levels(
            df_filtered, 
            df_interpolated,
            f'{self.lake_name} Water Level Changes ({begin_date} to {ending_date})'
        )
        
        # Create combined visualization if we have images
        if save_previews and not df_filtered.empty:
            combined_path = self.create_combined_visualization(df_filtered)
            if combined_path:
                print(f"Combined visualization saved to {combined_path}")
        
        print(f"Monitoring complete. Results saved to {self.results_dir}")
        if plot_path:
            print(f"Water level plot saved to {plot_path}")
        print(f"Sample satellite images saved to {self.images_dir}")
        print("\nTo embed the chart on a website:")
        print(f"- PNG: {self.results_dir}/{self.lake_name}_water_levels.png")
        print(f"- SVG: {self.results_dir}/{self.lake_name}_water_levels.svg")
        print(f"- Combined visualization: {self.results_dir}/{self.lake_name}_combined_visualization.svg")
        
        return df_filtered, df_interpolated


# Example usage
if __name__ == "__main__":
    client_id = ""#api client id here!
    client_secret= ""#api secret here!

    lake_talty_coords = [
        (21.489688755215752, 53.90419557585972),
		(21.479090855361903, 53.91016532222625),
		(21.473441390838587, 53.90626435415567),
		(21.47452722082508, 53.89664137732353),
		(21.481516459551074, 53.8851057348175),
		(21.49169227036299, 53.8771189015404),
        (21.49579760349613, 53.86396582268583),
        (21.501851707155424, 53.86237185780391),
        (21.51259312929008, 53.85699780053659),
        (21.517117340801377, 53.85827852748221),
        (21.530396699097338, 53.849671044844285),
        (21.546909206218288, 53.83975111713542),
        (21.54169196428319, 53.83301227402097),
        (21.54150425980211, 53.82534227542496),
        (21.55006209507866, 53.819404054699646),
        (21.555073049171995, 53.81923872318234),
		(21.561039195027377, 53.814825747255725),
		(21.55795040677708, 53.80428366705834),
		(21.563243751405167, 53.80046235931451),
		(21.574113944439688, 53.80276133707315),
		(21.57195938466046, 53.8112513984008),
		(21.575260716866865, 53.81776847101705),
		(21.562428359914946, 53.827768308773784),
		(21.565653119382688, 53.832754914931485),
		(21.567736512243528, 53.840010553504726),
		(21.562390950253615, 53.844129295198286),
		(21.55609016771203, 53.856558338693844),
		(21.548149225010377, 53.865781596566535),
		(21.535368094475018, 53.87422531763508),
		(21.524845855366834, 53.87618922222441),
		(21.518899437639448, 53.88270379224869),
		(21.513068249426226, 53.88502252241176),
		(21.51815980168979, 53.895688074579226),
		(21.51117412026781, 53.901530540352525),
		(21.507023045778567, 53.891692867640984),
		(21.50182307439232, 53.89352064443739),
		(21.507463391384988, 53.90090437054786),
		(21.503531391660147, 53.90457363405383),
		(21.489688755215752, 53.90419557585972)
    ]
    """(21.5580, 53.9244),
        (21.5627, 53.9187),
        (21.5668, 53.9132), 
        (21.5700, 53.9054),
        (21.5720, 53.8974),
        (21.5683, 53.8915),
        (21.5620, 53.8868),
        (21.5548, 53.8870),
        (21.5497, 53.8912),
        (21.5462, 53.8975),
        (21.5445, 53.9059),
        (21.5458, 53.9138),
        (21.5490, 53.9197),
        (21.5580, 53.9244)"""
    # Initialize lake monitor
    monitor = LakeMonitor(
        client_id=client_id,
        client_secret=client_secret,
        lake_name="Jezioro Tałty", 
        coordinates=lake_talty_coords
    )
    
    # Define date range for analysis
    # Using one year of data
    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now() - relativedelta(years=1)).strftime('%Y%m%d')
    
    # Run the monitoring pipeline
    # Limiting to 12 images (approximately monthly) to save resources
    filtered_df, interpolated_df = monitor.run_monitoring(
        start_date=start_date,
        end_date=end_date,
        cloud_percent=50,  # Higher threshold for initial query
        limit=12,  # Limit to ~1 image per month
        save_previews=True  # Save preview images
    )
    
    if not filtered_df.empty:
        print("\nStatistics for Lake Tałty water levels:")
        print(f"Mean water area: {filtered_df['water_area_km2'].mean():.2f} km²")
        print(f"Min water area: {filtered_df['water_area_km2'].min():.2f} km²")
        print(f"Max water area: {filtered_df['water_area_km2'].max():.2f} km²")
    else:
        print("\nNo valid data found for Lake Tałty. Please check search parameters and credentials.")