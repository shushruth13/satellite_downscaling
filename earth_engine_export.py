import ee
import geemap
import os
from datetime import datetime, timedelta
import geocoder
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='durable-destiny-456403-n8')

def get_city_coordinates(city_name, country_name):
    """Get coordinates for a city and create a bounding box around it"""
    try:
        # Create a geocoder instance
        geolocator = Nominatim(user_agent="no2_data_exporter")
        
        # Search for the city
        location = geolocator.geocode(f"{city_name}, {country_name}")
        
        if location is None:
            raise ValueError(f"Could not find coordinates for {city_name}, {country_name}")
        
        # Get the center coordinates
        center_lat = location.latitude
        center_lon = location.longitude
        
        # Create a bounding box (approximately 50km x 50km)
        # Move 25km in each direction
        north = geodesic(kilometers=25).destination((center_lat, center_lon), 0).latitude
        south = geodesic(kilometers=25).destination((center_lat, center_lon), 180).latitude
        east = geodesic(kilometers=25).destination((center_lat, center_lon), 90).longitude
        west = geodesic(kilometers=25).destination((center_lat, center_lon), 270).longitude
        
        return [west, south, east, north]
    except Exception as e:
        print(f"Error getting coordinates: {str(e)}")
        return None

def get_user_input():
    """Get region and date range from user input"""
    print("\n=== NO2 Data Export Configuration ===")
    
    # Get date range
    while True:
        try:
            start_date = input("\nEnter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            
            # Validate dates
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            
            if datetime.strptime(end_date, '%Y-%m-%d') < datetime.strptime(start_date, '%Y-%m-%d'):
                print("Error: End date must be after start date")
                continue
            break
        except ValueError:
            print("Error: Please enter dates in YYYY-MM-DD format")
    
    # Get city and country
    while True:
        try:
            print("\nEnter location details:")
            city_name = input("City name: ")
            country_name = input("Country name: ")
            
            # Get coordinates for the city
            region = get_city_coordinates(city_name, country_name)
            if region is None:
                print("Error: Could not find coordinates for the specified location")
                continue
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return start_date, end_date, region, f"{city_name}, {country_name}"

def generate_child_friendly_analysis(no2_level):
    """
    Generate child-friendly analysis of NO2 levels
    
    Parameters:
    - no2_level: The mean NO2 concentration value
    """
    # Define NO2 level categories with detailed explanations
    if no2_level < 0.0001:
        level = "very good"
        emoji = "🌱"
        message = "The air is very clean! It's perfect for playing outside."
        causes = "This area has very little pollution because there are fewer cars and factories."
        control = "Keep up the good work! Continue using clean energy and public transport."
    elif no2_level < 0.0002:
        level = "good"
        emoji = "😊"
        message = "The air is good! You can play outside safely."
        causes = "The air is clean, but there might be some cars and buses nearby."
        control = "Try to walk or bike more often, and plant more trees in your neighborhood."
    elif no2_level < 0.0003:
        level = "moderate"
        emoji = "⚠️"
        message = "The air is okay, but it's better to play inside today."
        causes = "There are more cars and factories in this area, making the air a bit dirty."
        control = "We need to use fewer cars and make sure factories follow clean air rules."
    else:
        level = "poor"
        emoji = "😷"
        message = "The air is not good today. It's best to stay inside."
        causes = "This area has lots of traffic and industrial activities, making the air very dirty."
        control = "We need to take immediate action to reduce pollution and use cleaner energy."

    analysis = f"""
🌟 Air Quality Report 🌟

{emoji} Today's Air Quality: {level.upper()}
{message}

📝 What is NO2?
NO2 is like invisible smoke that comes from cars and factories. It's not good for our lungs!

🔍 Why is it important?
- It helps us know if the air is safe to breathe
- It tells us if we can play outside
- It helps us take care of our planet

❓ What's making the air dirty in this area?
{causes}

💡 How can we help make the air better?
1. Walk or bike instead of using cars
2. Plant trees and flowers
3. Save energy by turning off lights
4. Tell grown-ups to use less electricity
5. Use public transportation
6. Support clean energy projects

🎯 What needs to be done?
{control}

🌍 Fun Facts:
- Trees are like nature's air filters!
- One tree can clean the air for 4 people
- Walking to school helps keep the air clean
- Electric cars make less pollution

Remember: Clean air means happy lungs! 🌈
"""
    return analysis

def export_no2_data(start_date, end_date, region, location_name, output_dir='./data'):
    """
    Export NO2 data from Google Earth Engine
    
    Parameters:
    - start_date: Start date in 'YYYY-MM-DD' format
    - end_date: End date in 'YYYY-MM-DD' format
    - region: Region of interest as [min_lon, min_lat, max_lon, max_lat]
    - location_name: Name of the location (city, country)
    - output_dir: Directory to save the exported data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the region of interest
    roi = ee.Geometry.Rectangle(region)
    
    print("\nFetching NO2 data...")
    # Get the NO2 dataset
    no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2') \
        .select('tropospheric_NO2_column_number_density') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    # Calculate mean for the time period
    mean_no2 = no2.mean()
    
    # Get the mean NO2 value
    no2_value = mean_no2.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=1000
    ).get('tropospheric_NO2_column_number_density').getInfo()
    
    # Generate child-friendly analysis
    analysis = generate_child_friendly_analysis(no2_value)
    
    # Create a safe filename from location name
    safe_location = location_name.replace(",", "_").replace(" ", "_")
    
    # Export the image
    task = ee.batch.Export.image.toDrive(
        image=mean_no2,
        description=f'NO2_Data_{safe_location}_{start_date}_{end_date}',
        scale=1000,  # 1km resolution
        region=roi,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    
    # Start the export task
    task.start()
    
    print(f"\nExport task started. Task ID: {task.id}")
    print("Please check your Google Drive for the exported file.")
    print(f"The file will be named: NO2_Data_{safe_location}_{start_date}_{end_date}.tif")
    
    # Print the child-friendly analysis
    print("\n" + analysis)

if __name__ == "__main__":
    print("Welcome to the NO2 Data Export Tool!")
    print("This tool will help you export NO2 concentration data from Google Earth Engine.")
    
    # Get user input
    start_date, end_date, region, location_name = get_user_input()
    
    # Confirm the selection
    print("\nPlease confirm your selection:")
    print(f"Location: {location_name}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Region: Longitude {region[0]:.4f} to {region[2]:.4f}, Latitude {region[1]:.4f} to {region[3]:.4f}")
    
    confirm = input("\nProceed with export? (y/n): ").lower()
    if confirm == 'y':
        export_no2_data(start_date, end_date, region, location_name)
    else:
        print("Export cancelled.") 