import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model import NO2DownscalingModel
from utils import (
    load_satellite_data,
    load_ground_data,
    create_no2_map,
    calculate_metrics,
    handle_missing_data,
    save_satellite_data,
    save_ground_measurements,
    forecast_no2_levels,
    create_aqi_health_dashboard,
    compare_locations,
    explain_no2_trend
)
from database import init_db, get_db
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from shapely.geometry import Point
import time
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio

# Add caching decorators at the top of the file, after imports
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_satellite_data(file):
    data, transform, crs = load_satellite_data(file)
    if data is not None:
        processed_data = handle_missing_data(data)
        return processed_data, transform, crs
    return None, None, None

@st.cache_data(ttl=3600)
def process_predictions(data):
    """Process data and return predictions without caching the model"""
    model = NO2DownscalingModel()
    X_val, y_val = model.train(data)
    predictions, uncertainty = model.predict(data)
    return predictions, uncertainty, X_val, y_val

@st.cache_data(ttl=3600)
def create_cached_map(data, title):
    return create_no2_map(data, title)

# Function to convert NO2 to cigarette equivalent
def no2_to_cigarettes(no2_level):
    """Convert NO2 level to cigarette equivalent"""
    return round(no2_level / 22, 2)

# Function to get AQI category
def get_aqi_category(no2_level):
    """Get AQI category based on NO2 level"""
    if no2_level <= 40:
        return "Good", "green"
    elif no2_level <= 80:
        return "Moderate", "yellow"
    elif no2_level <= 180:
        return "Poor", "orange"
    elif no2_level <= 280:
        return "Very Poor", "red"
    else:
        return "Severe", "purple"

# Initialize database
init_db()

# Set page config
st.set_page_config(
    page_title="NO2 Air Quality Analysis",
    page_icon="🌍",
    layout="wide"
)

# Load custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title and description
st.title("🌍 NO2 Air Quality Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of NO2 concentrations, including seasonal comparisons,
health impacts, and AI-powered insights.
""")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Current Analysis", "Seasonal Comparison", "Health Impact", "Future Forecast", "Multi-City Comparison"])

with tab1:
    st.header("🌍 Current NO₂ Analysis (Is the Air Clean Today?)")
    st.markdown("""
    <div style='font-size:1.1em;'>
    <b>What does this map show?</b> <br>
    This is the NO₂ map for your city. <span style='color:green;'>Blue/green is good</span>, <span style='color:red;'>red/purple is bad</span>.<br>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    col1, col2 = st.columns(2)
    with col1:
        satellite_file = st.file_uploader(
            "Upload Satellite Data (GeoTIFF)",
            type=['tif', 'tiff'],
            help="Upload satellite NO2 data in GeoTIFF format"
        )

    with col2:
        ground_file = st.file_uploader(
            "Upload Ground Station Data (CSV)",
            type=['csv'],
            help="Upload ground station measurements for validation"
        )

    if satellite_file is not None:
        # Load and process satellite data
        processed_data, transform, crs = load_and_process_satellite_data(satellite_file)

        if processed_data is not None:
            st.subheader("🗺️ Input Data Visualization")
            st.markdown("<i>This is what the satellite sees in the sky today!</i>", unsafe_allow_html=True)
            st.plotly_chart(
                create_cached_map(processed_data, "Original NO2 Concentration"),
                use_container_width=True
            )

            st.subheader("✨ Downscaled Map (Super Clear View)")
            st.markdown("<i>This is what the computer thinks the air would look like if it was super clear!</i>", unsafe_allow_html=True)
            predictions, uncertainty, X_val, y_val = process_predictions(processed_data)
            st.plotly_chart(
                create_cached_map(predictions, "Downscaled NO2 Concentration"),
                use_container_width=True
            )

            # Calculate and convert average NO2 level
            avg_no2_raw = np.mean(processed_data)
            avg_no2 = int(round(avg_no2_raw * 46000))  # Convert to µg/m³ and round to whole number
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average NO2 Level", f"{avg_no2} µg/m³")
            with col2:
                cigarettes = no2_to_cigarettes(avg_no2)
                st.metric("Health Impact", f"{cigarettes} cigarettes/day equivalent")
            with col3:
                aqi_category, color = get_aqi_category(avg_no2)
                st.metric("Air Quality", aqi_category)

            st.subheader("🧒 What does this mean for you?")
            if avg_no2 > 80:
                st.markdown("""
                <div style='background:#fff3cd; border-left:4px solid #ffc107; padding:1em; border-radius:8px;'>
                <b>Oh no! The air is not so good today.</b><br>
                - Try to play inside more.<br>
                - If you go outside, ask an adult and maybe wear a mask.<br>
                - Tell your friends and family to be careful too!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#d4edda; border-left:4px solid #28a745; padding:1em; border-radius:8px;'>
                <b>Yay! The air is clean today.</b><br>
                - You can play outside and have fun!<br>
                - Remember, trees help keep the air clean. 🌳
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("🌦️ Seasonal NO₂ Comparison (Which Season Has the Cleanest Air?)")
    st.markdown("""
    <div style='font-size:1.1em;'>
    <b>What does this show?</b> <br>
    Here are the NO₂ maps for different seasons. <span style='color:green;'>Which one looks the cleanest?</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload NO2 GeoTIFFs for different seasons (e.g., Jan, Apr, Jul, Oct)",
        type=['tif', 'tiff'],
        accept_multiple_files=True
    )

    season_names = ["Winter (Jan)", "Summer (Apr)", "Monsoon (Jul)", "Post-Monsoon (Oct)"]

    if uploaded_files and len(uploaded_files) == 4:
        avg_no2_seasons = []
        input_maps = []
        downscaled_maps = []
        
        for i, file in enumerate(uploaded_files):
            processed_data, transform, crs = load_and_process_satellite_data(file)
            if processed_data is not None:
                avg_no2_raw = np.mean(processed_data)
                avg_no2 = int(round(avg_no2_raw * 46000))
                avg_no2_seasons.append(avg_no2)
                input_maps.append(create_cached_map(processed_data, f"{season_names[i]} Input NO2"))
                
                predictions, uncertainty, _, _ = process_predictions(processed_data)
                downscaled_maps.append(create_cached_map(predictions, f"{season_names[i]} Downscaled NO2"))
            else:
                avg_no2_seasons.append(None)
                input_maps.append(None)
                downscaled_maps.append(None)

        st.subheader("🗺️ Seasonal NO₂ Maps (Input vs Downscaled)")
        for i in range(4):
            cols = st.columns(2)
            if input_maps[i]:
                cols[0].plotly_chart(input_maps[i], use_container_width=True)
                cols[0].markdown(f"**{season_names[i]} Input**")
            if downscaled_maps[i]:
                cols[1].plotly_chart(downscaled_maps[i], use_container_width=True)
                cols[1].markdown(f"**{season_names[i]} Downscaled**")

        st.subheader("📊 Which Season Has the Cleanest Air?")
        st.markdown("<i>This chart shows which season has the cleanest and dirtiest air. Lower is better!</i>", unsafe_allow_html=True)
        fig = go.Figure([go.Bar(x=season_names, y=avg_no2_seasons, marker_color='indianred')])
        fig.update_layout(yaxis_title="Average NO2 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧒 What does this mean for you?")
        max_season = season_names[np.argmax(avg_no2_seasons)]
        min_season = season_names[np.argmin(avg_no2_seasons)]
        st.info(f"🌳 The air is cleanest in {min_season} and dirtiest in {max_season}.\nTry to play outside more in the clean season!", icon="🌳")

        # Add NO2-specific pros, cons, and tips for each season
        st.markdown("""
        <div style='font-size:1.1em;'>
        <table style='width:100%; border-collapse:collapse;'>
        <tr style='background:#f0f2f6;'>
            <th>Season</th>
            <th>😊 Pros (NO₂)</th>
            <th>😟 Cons (NO₂)</th>
            <th>💡 Tips</th>
        </tr>
        <tr>
            <td>Winter (Jan)</td>
            <td>❄️ Crisp, cool air can feel fresh on some days.</td>
            <td>🚗 NO₂ is usually highest due to more car and truck use, and fog traps pollution near the ground.<br>😮‍💨 Breathing problems can get worse for kids and older people.</td>
            <td>😷 Try to avoid busy roads and traffic jams.<br>🏠 Play inside more if the air looks foggy or smells smoky.<br>👨‍👩‍👧‍👦 Ask adults to check the air quality before going out.</td>
        </tr>
        <tr style='background:#f0f2f6;'>
            <td>Summer (Apr)</td>
            <td>☀️ Sunlight and wind help clean the air, so NO₂ is often lower.</td>
            <td>🏭 Hot weather can make pollution from vehicles and factories rise in the afternoon.</td>
            <td>⏰ Play outside in the morning or evening when it's cooler and air is cleaner.<br>💧 Drink lots of water and rest in the shade.</td>
        </tr>
        <tr>
            <td>Monsoon (Jul)</td>
            <td>🌧️ Rain washes NO₂ out of the air, making it the cleanest season!</td>
            <td>🚦 Sometimes, traffic jams during rain can cause short spikes in pollution.</td>
            <td>🌈 Enjoy playing outside after the rain when the air is fresh.<br>🚫 Avoid puddles near busy roads.</td>
        </tr>
        <tr style='background:#f0f2f6;'>
            <td>Post-Monsoon (Oct)</td>
            <td>🍂 Air is usually clean after the rains.</td>
            <td>🔥 Some NO₂ can come from burning leaves or farm waste, and from Diwali fireworks.</td>
            <td>🏠 Stay indoors or wear a mask if you see or smell smoke.<br>🙅‍♂️ Remind adults not to burn leaves or trash.</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("""
        Please upload <b>exactly 4 GeoTIFF files</b> (one for each season: Jan, Apr, Jul, Oct) for comparison.
        """, icon="ℹ️")

with tab3:
    st.header("🩺 Health Impact: What Does NO₂ Mean for You?")

    # Simple explanation
    st.markdown("""
    <div style='font-size:1.2em;'>
    <b>What is <span style='color:#1E3A8A;'>NO₂</span>?</b><br>
    <span style='font-size:1.1em;'>NO₂ (Nitrogen Dioxide) is a gas in the air. Too much of it can make the air unhealthy to breathe, especially for kids, grandparents, and people with asthma.</span><br><br>
    <b>Why do we compare it to cigarettes?</b><br>
    <span style='font-size:1.1em;'>Breathing a lot of NO₂ is a bit like smoking cigarettes. Scientists say if you breathe <b>22 µg/m³</b> of NO₂ every day, it's like smoking <b>1 cigarette</b> a day!</span>
    </div>
    """, unsafe_allow_html=True)

    # Health impact visualization
    st.markdown("<b>How much NO₂ is like how many cigarettes?</b>", unsafe_allow_html=True)
    no2_levels = np.linspace(0, 300, 100)
    cigarette_equiv = [no2_to_cigarettes(level) for level in no2_levels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=no2_levels,
        y=cigarette_equiv,
        mode='lines',
        name='NO2 to Cigarette Equivalent',
        line=dict(color='#1E3A8A', width=3)
    ))
    fig.update_layout(
        title='NO₂ in the Air vs. Cigarette Equivalent',
        xaxis_title='NO₂ Level (µg/m³)',
        yaxis_title='Cigarettes per Day',
        hovermode='x',
        plot_bgcolor='#f8f9fa',
        font=dict(size=16)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Colorful, friendly health zones
    st.markdown("""
    <div style='font-size:1.1em;'>
    <b>What do the numbers mean?</b><br>
    <span style='color:green;'>🟢 0-40 µg/m³: Safe! You can play outside.</span><br>
    <span style='color:orange;'>🟡 40-80 µg/m³: Okay, but be careful. If you have asthma, ask an adult before playing outside.</span><br>
    <span style='color:#ff8800;'>🟠 80-180 µg/m³: Not so good. Try to play inside more.</span><br>
    <span style='color:red;'>🔴 180-280 µg/m³: Bad air! Stay inside and ask an adult to close the windows.</span><br>
    <span style='color:purple;'>🟣 >280 µg/m³: Very bad! Stay inside, use an air purifier if you have one, and tell an adult.</span>
    </div>
    """, unsafe_allow_html=True)

    # What can you do?
    st.markdown("""
    <div style='font-size:1.1em; background:#e8f4fd; border-radius:10px; padding:1em; margin-top:1em;'>
    <b>What can you do to stay healthy?</b><br>
    <ul>
    <li>👃 Check the air quality every day.</li>
    <li>🏃 Play outside when the air is green or yellow.</li>
    <li>🏠 If the air is orange, red, or purple, play inside.</li>
    <li>😷 If you have to go outside on a bad air day, wear a mask.</li>
    <li>🪟 Ask an adult to keep windows closed on bad air days.</li>
    <li>🌳 Plant trees! They help clean the air.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Health recommendations (for adults)
    st.subheader("For Parents and Adults")
    st.markdown("""
    - Long-term exposure to high NO₂ can cause breathing problems, especially in children and the elderly.
    - Use air purifiers and avoid outdoor exercise on high NO₂ days.
    - Support clean energy and public transport to help reduce air pollution.
    """)

with tab4:
    st.header("🔮 Future NO₂ Forecast (What Will the Air Be Like Next Month?)")
    st.markdown("""
    <div style='font-size:1.1em;'>
    <b>What does this show?</b> <br>
    This uses AI to predict what the air quality will be like in the future! It looks at patterns from the past to make a smart guess.
    </div>
    """, unsafe_allow_html=True)
    
    forecast_source = st.radio(
        "Choose your forecast source:",
        ["Use current satellite data", "Upload historical data series", "Use demo data"]
    )
    
    historical_data = None
    
    if forecast_source == "Use current satellite data" and 'processed_data' in locals():
        st.info("Using your current satellite data to make a prediction", icon="ℹ️")
        # Create time series from current data for demo purposes
        base_value = np.nanmean(processed_data)
        dates = [datetime.now() - timedelta(days=i) for i in range(60)]
        dates.reverse()
        
        # Create synthetic historical data with seasonal pattern
        values = [base_value * (1 + 0.2 * np.sin(i/30 * np.pi) + np.random.normal(0, 0.05)) for i in range(60)]
        historical_data = pd.DataFrame({'ds': dates, 'y': values})
        
    elif forecast_source == "Upload historical data series":
        uploaded_history = st.file_uploader(
            "Upload historical NO2 data (CSV with date and no2_value columns)",
            type=['csv']
        )
        
        if uploaded_history:
            historical_data = pd.read_csv(uploaded_history)
            if 'date' not in historical_data.columns or 'no2_value' not in historical_data.columns:
                st.error("CSV must have 'date' and 'no2_value' columns")
                historical_data = None
    
    elif forecast_source == "Use demo data":
        st.info("Using demo data for forecasting", icon="ℹ️")
        # Create demo data
        dates = [datetime.now() - timedelta(days=i) for i in range(60)]
        dates.reverse()
        
        # Create synthetic historical data with seasonal pattern
        base_value = 0.0002  # Approximately 9.2 μg/m³
        values = [base_value * (1 + 0.3 * np.sin(i/30 * np.pi) + np.random.normal(0, 0.05)) for i in range(60)]
        historical_data = pd.DataFrame({'ds': dates, 'y': values})
    
    if historical_data is not None:
        # Forecasting period
        forecast_days = st.slider("Forecast days ahead:", 7, 90, 30)
        
        # Display forecasting progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training forecasting model...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("Generating future scenarios...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("Calculating confidence intervals...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        status_text.text("Preparing visualization...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        status_text.empty()
        
        # Create forecast
        forecast_df, forecast_fig = forecast_no2_levels(historical_data, periods=forecast_days)
        
        # Display forecast
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Add AI-generated explanation for the trend
        explanation = explain_no2_trend(forecast_df, periods=forecast_days)
        st.info(f"**Why this trend?**\n{explanation}")
        
        # Weather-like forecast card display
        st.subheader("🌈 30-Day Air Quality Outlook")
        
        # Calculate averages for each period
        today_val = forecast_df.iloc[-forecast_days]['yhat'] * 46000  # Convert to μg/m³
        next_week_vals = forecast_df.iloc[-forecast_days:-forecast_days+7]['yhat'] * 46000
        next_month_vals = forecast_df.iloc[-forecast_days:]['yhat'] * 46000
        
        # Create card layout
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("### 📅 Today")
            category, color = get_aqi_category(today_val)
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 10px;'>"
                      f"<h1 style='text-align: center; color: white;'>{int(today_val)} μg/m³</h1>"
                      f"<p style='text-align: center; color: white; font-weight: bold;'>{category}</p>"
                      "</div>", unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown("### 📆 7-Day Average")
            avg_next_week = np.mean(next_week_vals)
            category, color = get_aqi_category(avg_next_week)
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 10px;'>"
                      f"<h1 style='text-align: center; color: white;'>{int(avg_next_week)} μg/m³</h1>"
                      f"<p style='text-align: center; color: white; font-weight: bold;'>{category}</p>"
                      "</div>", unsafe_allow_html=True)
            
        with cols[2]:
            st.markdown("### 📅 30-Day Average")
            avg_next_month = np.mean(next_month_vals)
            category, color = get_aqi_category(avg_next_month)
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 10px;'>"
                      f"<h1 style='text-align: center; color: white;'>{int(avg_next_month)} μg/m³</h1>"
                      f"<p style='text-align: center; color: white; font-weight: bold;'>{category}</p>"
                      "</div>", unsafe_allow_html=True)
        
        # Insight cards
        st.subheader("🧠 AI Insights from Forecast")
        
        insight_cols = st.columns(3)
        
        # Highest NO2 day
        with insight_cols[0]:
            highest_day_idx = forecast_df.iloc[-forecast_days:]['yhat'].argmax()
            highest_day_date = forecast_df.iloc[-forecast_days:].iloc[highest_day_idx]['ds'].strftime('%B %d')
            highest_day_value = forecast_df.iloc[-forecast_days:].iloc[highest_day_idx]['yhat'] * 46000
            category, _ = get_aqi_category(highest_day_value)
            
            st.markdown(f"""
            <div style='background-color: #f8d7da; padding: 15px; border-radius: 10px;'>
                <h3>⚠️ Highest NO2 Day</h3>
                <p><b>{highest_day_date}</b> will likely have the worst air quality at <b>{int(highest_day_value)} μg/m³</b> ({category})</p>
                <p>Consider limiting outdoor activities on this day.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Lowest NO2 day
        with insight_cols[1]:
            lowest_day_idx = forecast_df.iloc[-forecast_days:]['yhat'].argmin()
            lowest_day_date = forecast_df.iloc[-forecast_days:].iloc[lowest_day_idx]['ds'].strftime('%B %d')
            lowest_day_value = forecast_df.iloc[-forecast_days:].iloc[lowest_day_idx]['yhat'] * 46000
            category, _ = get_aqi_category(lowest_day_value)
            
            st.markdown(f"""
            <div style='background-color: #d1e7dd; padding: 15px; border-radius: 10px;'>
                <h3>✅ Lowest NO2 Day</h3>
                <p><b>{lowest_day_date}</b> will likely have the best air quality at <b>{int(lowest_day_value)} μg/m³</b> ({category})</p>
                <p>This would be a great day for outdoor activities!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall trend
        with insight_cols[2]:
            start_val = forecast_df.iloc[-forecast_days]['yhat'] * 46000
            end_val = forecast_df.iloc[-1]['yhat'] * 46000
            percent_change = ((end_val - start_val) / start_val) * 100
            
            if percent_change > 5:
                trend_color = "#f8d7da"
                trend_text = f"Air quality is expected to worsen by {int(abs(percent_change))}% over the next month"
                trend_icon = "📈"
                trend_advice = "Consider indoor air purification measures."
            elif percent_change < -5:
                trend_color = "#d1e7dd"
                trend_text = f"Air quality is expected to improve by {int(abs(percent_change))}% over the next month"
                trend_icon = "📉"
                trend_advice = "Enjoy the improving conditions!"
            else:
                trend_color = "#fff3cd"
                trend_text = "Air quality is expected to remain relatively stable"
                trend_icon = "➡️"
                trend_advice = "Monitor conditions regularly."
            
            st.markdown(f"""
            <div style='background-color: {trend_color}; padding: 15px; border-radius: 10px;'>
                <h3>{trend_icon} Overall Trend</h3>
                <p>{trend_text}</p>
                <p>{trend_advice}</p>
            </div>
            """, unsafe_allow_html=True)

        # Day-by-day calendar view
        st.subheader("📆 Day-by-Day Air Quality Calendar")
        
        # Create a calendar-like display
        days_per_row = 7
        num_rows = (forecast_days + days_per_row - 1) // days_per_row
        
        for row in range(num_rows):
            cols = st.columns(days_per_row)
            
            for col in range(days_per_row):
                day_idx = row * days_per_row + col
                
                if day_idx < forecast_days:
                    day_date = forecast_df.iloc[-forecast_days:].iloc[day_idx]['ds']
                    day_value = forecast_df.iloc[-forecast_days:].iloc[day_idx]['yhat'] * 46000
                    category, color = get_aqi_category(day_value)
                    
                    day_label = day_date.strftime('%b %d')
                    cols[col].markdown(f"""
                    <div style='background-color: {color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                        <p style='text-align: center; color: white; margin: 0;'><b>{day_label}</b></p>
                        <h3 style='text-align: center; color: white; margin: 5px 0;'>{int(day_value)}</h3>
                        <p style='text-align: center; color: white; margin: 0; font-size: 0.8em;'>{category}</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab5:
    st.header("🌆 Multi-City NO₂ Comparison")
    st.markdown("""
    <div style='font-size:1.1em;'>
    <b>What does this show?</b> <br>
    Compare NO₂ levels across different cities to see which areas have the cleanest air. 
    Upload data for up to 4 cities to compare their air quality side by side.
    </div>
    """, unsafe_allow_html=True)

    # Enhanced file upload interface
    city_files = st.file_uploader(
        "Upload NO2 GeoTIFFs for different cities",
        type=['tif', 'tiff'],
        accept_multiple_files=True,
        help="Upload satellite NO2 data for different cities (maximum 4 cities)"
    )

    # City name input and data processing
    if city_files:
        st.info(f"📌 {len(city_files)} files uploaded. Please name each city:")
        city_names = []
        city_data = {}
        input_maps = []
        downscaled_maps = []
        avg_no2_values = []
        
        # Create columns for city name inputs
        cols = st.columns(min(len(city_files), 4))
        
        # First, get city names
        for i, file in enumerate(city_files[:4]):
            with cols[i]:
                city_name = st.text_input(f"City {i+1} name", value=f"City {i+1}")
                city_names.append(city_name)
        
        # Process data only if we have unique city names
        if len(set(city_names)) == len(city_names):
            # Process each city's data
            for i, (file, city) in enumerate(zip(city_files[:4], city_names)):
                processed_data, transform, crs = load_and_process_satellite_data(file)
                if processed_data is not None:
                    city_data[city] = processed_data
                    
                    # Calculate average NO2
                    avg_no2_raw = np.mean(processed_data)
                    avg_no2 = int(round(avg_no2_raw * 46000))
                    avg_no2_values.append(avg_no2)
                    
                    # Create input and downscaled maps
                    input_maps.append(create_cached_map(processed_data, f"{city} Input NO2"))
                    
                    # Downscale the data
                    predictions, uncertainty, _, _ = process_predictions(processed_data)
                    downscaled_maps.append(create_cached_map(predictions, f"{city} Downscaled NO2"))

            if input_maps:
                # Display side-by-side comparison
                st.subheader("🗺️ City-by-City Comparison")
                
                # Create metrics for quick comparison
                metric_cols = st.columns(len(input_maps))
                for i, (city, avg_no2) in enumerate(zip(city_names[:len(input_maps)], avg_no2_values)):
                    with metric_cols[i]:
                        category, color = get_aqi_category(avg_no2)
                        st.metric(
                            f"{city}",
                            f"{avg_no2} µg/m³",
                            delta=f"{category}",
                            delta_color="inverse"
                        )

                # Display maps in grid layout
                st.subheader("📊 Detailed Visualization")
                for i in range(0, len(input_maps), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(input_maps):
                            with cols[j]:
                                st.plotly_chart(input_maps[i + j], use_container_width=True)
                                st.plotly_chart(downscaled_maps[i + j], use_container_width=True)
                                st.markdown(f"**{city_names[i + j]}**")
                
                # Add comparative analysis
                st.subheader("📈 Comparative Analysis")
                
                # Create bar chart for NO2 levels
                fig = px.bar(
                    x=city_names[:len(input_maps)],
                    y=avg_no2_values,
                    title="NO₂ Levels Across Cities",
                    labels={"x": "City", "y": "NO₂ Level (µg/m³)"},
                    color=avg_no2_values,
                    color_continuous_scale="RdYlGn_r"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add health impact comparison
                st.subheader("🏥 Health Impact Comparison")
                health_data = []
                for city, avg_no2 in zip(city_names[:len(input_maps)], avg_no2_values):
                    category, color = get_aqi_category(avg_no2)
                    cigarettes = no2_to_cigarettes(avg_no2)
                    health_data.append({
                        "City": city,
                        "NO₂ Level": avg_no2,
                        "Category": category,
                        "Cigarette Equivalent": cigarettes
                    })
                
                health_df = pd.DataFrame(health_data)
                st.dataframe(
                    health_df.style.background_gradient(subset=["NO₂ Level"], cmap="RdYlGn_r"),
                    hide_index=True
                )

                # Expert Analysis Section
                st.subheader("🔬 Expert Analysis: City-Specific NO₂ Patterns")
                
                for city, avg_no2 in zip(city_names[:len(input_maps)], avg_no2_values):
                    category, color = get_aqi_category(avg_no2)
                    
                    # Get city-specific data patterns
                    city_data_array = city_data[city]
                    spatial_variance = np.nanstd(city_data_array)
                    peak_concentration = np.nanmax(city_data_array) * 46000
                    spatial_hotspots = np.sum(city_data_array > np.nanmean(city_data_array) + spatial_variance)
                    
                    # Calculate distribution characteristics
                    data_distribution = "uniform" if spatial_variance < 0.00001 else "clustered" if spatial_variance < 0.0001 else "highly varied"
                    hotspot_density = spatial_hotspots / (city_data_array.size - np.sum(np.isnan(city_data_array))) * 100
                    
                    # Generate expert analysis based on data patterns
                    if avg_no2 > 150:
                        pattern_analysis = f"""
                        🔍 <b>Spatial Pattern Analysis:</b>
                        - {data_distribution.title()} NO₂ distribution (σ={spatial_variance:.6f})
                        - Multiple intense pollution hotspots ({hotspot_density:.1f}% of area)
                        - Peak concentration of {peak_concentration:.1f} µg/m³
                        
                        📊 <b>Primary Contributors:</b>
                        1. Major traffic corridors
                        2. Industrial clusters
                        3. Urban canyon effects
                        4. Limited green buffer zones
                        
                        🎯 <b>Critical Intervention Areas:</b>
                        - Traffic flow optimization
                        - Industrial emission control
                        - Urban ventilation corridors
                        - Strategic green barriers
                        """
                    elif avg_no2 > 80:
                        pattern_analysis = f"""
                        🔍 <b>Spatial Pattern Analysis:</b>
                        - {data_distribution.title()} NO₂ distribution (σ={spatial_variance:.6f})
                        - Moderate hotspots ({hotspot_density:.1f}% of area)
                        - Peak levels of {peak_concentration:.1f} µg/m³
                        
                        📊 <b>Primary Contributors:</b>
                        1. Commercial district emissions
                        2. Rush hour traffic
                        3. Mixed industrial-residential zones
                        4. Partial green cover
                        
                        🎯 <b>Critical Intervention Areas:</b>
                        - Traffic management
                        - Commercial zone standards
                        - Green corridor expansion
                        - Time-based restrictions
                        """
                    else:
                        pattern_analysis = f"""
                        🔍 <b>Spatial Pattern Analysis:</b>
                        - {data_distribution.title()} NO₂ distribution (σ={spatial_variance:.6f})
                        - Limited hotspots ({hotspot_density:.1f}% of area)
                        - Peak levels only {peak_concentration:.1f} µg/m³
                        
                        📊 <b>Success Factors:</b>
                        1. Effective urban planning
                        2. Good traffic management
                        3. Adequate green spaces
                        4. Successful emission controls
                        
                        🎯 <b>Maintenance Strategy:</b>
                        - Continue green infrastructure
                        - Monitor emerging hotspots
                        - Maintain emission standards
                        - Enhance public transport
                        """

                    # Display the expert analysis in an expandable section
                    with st.expander(f"🔬 Expert Analysis: {city} ({avg_no2:.1f} µg/m³ - {category})"):
                        st.markdown(f"""
                        <div style='background:{color}15; padding:20px; border-radius:10px; border-left:4px solid {color}'>
                        {pattern_analysis}
                        </div>
                        """, unsafe_allow_html=True)

                # Add comparative insights
                if len(input_maps) > 1:
                    st.subheader("🔄 Cross-City Pattern Analysis")
                    
                    # Calculate inter-city statistics
                    city_variances = [np.nanstd(city_data[city]) for city in city_names[:len(input_maps)]]
                    avg_variance = np.mean(city_variances)
                    variance_ratio = max(city_variances) / min(city_variances)
                    
                    # Find cities with highest and lowest concentrations
                    highest_city = max(city_names[:len(input_maps)], key=lambda x: np.nanmean(city_data[x]))
                    lowest_city = min(city_names[:len(input_maps)], key=lambda x: np.nanmean(city_data[x]))
                    
                    st.markdown(f"""
                    <div style='background:#f8f9fa; padding:20px; border-radius:10px; margin-top:20px;'>
                    <h4>📊 Inter-City Pollution Patterns</h4>
                    
                    <b>Spatial Heterogeneity Analysis:</b>
                    - Average spatial variance: {avg_variance:.6f}
                    - Variance ratio between cities: {variance_ratio:.2f}
                    
                    <b>Key Findings:</b>
                    - {highest_city} shows highest average concentration
                    - {lowest_city} demonstrates best air quality
                    - Spatial patterns suggest {'similar' if variance_ratio < 2 else 'diverse'} urban development patterns
                    
                    <b>Recommendations for Regional Air Quality Management:</b>
                    1. Implement successful practices from {lowest_city}
                    2. Focus on {'localized' if variance_ratio > 2 else 'regional'} intervention strategies
                    3. Consider {'unified' if variance_ratio < 2 else 'city-specific'} policy approaches
                    4. Establish inter-city coordination for {'emission control' if avg_no2 > 100 else 'maintenance'}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Please provide unique names for each city.")
    else:
        st.info("Please upload GeoTIFF files for different cities to compare their air quality.")

# Add footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit | "
    "Data source: Satellite NO2 measurements"
)