import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import SatelliteData, GroundMeasurement
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter

def load_satellite_data(file_path):
    """Load and preprocess satellite data."""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Apply floor function to replace any negative values with zero
            if data is not None:
                data = np.maximum(data, 0)
                
            return data, transform, crs
    except Exception as e:
        return None, None, None

def save_satellite_data(db: Session, data, transform, timestamp=None):
    """Save satellite data to database."""
    if timestamp is None:
        timestamp = datetime.utcnow()

    rows, cols = data.shape
    resolution = transform[0]  # pixel size in coordinate system units

    for i in range(rows):
        for j in range(cols):
            if not np.isnan(data[i, j]):
                lon, lat = transform * (j, i)
                db_entry = SatelliteData(
                    timestamp=timestamp,
                    latitude=lat,
                    longitude=lon,
                    no2_value=float(data[i, j]),
                    resolution=resolution,
                    source='satellite'
                )
                db.add(db_entry)

    db.commit()

def load_ground_data(file_path):
    """Load ground station measurement data."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        return None

def save_ground_measurements(db: Session, data):
    """Save ground measurements to database."""
    for _, row in data.iterrows():
        measurement = GroundMeasurement(
            timestamp=datetime.utcnow(),
            latitude=row['latitude'],
            longitude=row['longitude'],
            no2_value=row['no2_value'],
            station_name=row.get('station_name', 'unknown')
        )
        db.add(measurement)
    db.commit()

def create_no2_map(data, title="NO2 Concentration Map"):
    """Create an interactive map visualization using plotly."""
    fig = px.imshow(
        data,
        labels=dict(color="NO2 (μg/m³)"),
        title=title,
        color_continuous_scale="RdYlBu_r"
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def handle_missing_data(data, method='knn'):
    """
    Handle missing or invalid data points using advanced imputation techniques.
    
    Parameters:
    - data: Input data array
    - method: Imputation method ('knn' or 'mice')
    
    Returns:
    - processed_data: Imputed data array
    """
    # Convert to numpy array if not already
    data = np.array(data)
    
    # Create spatial features for better imputation
    rows, cols = data.shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    # Stack the data with spatial coordinates
    spatial_data = np.stack([
        data,
        x_coords / cols,  # Normalized x coordinates
        y_coords / rows   # Normalized y coordinates
    ], axis=-1)
    
    # Reshape for imputation
    n_samples = rows * cols
    reshaped_data = spatial_data.reshape(n_samples, 3)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reshaped_data)
    
    if method == 'knn':
        # Use KNN imputation with spatial context
        imputer = KNNImputer(
            n_neighbors=5,
            weights='uniform',
            metric='nan_euclidean'
        )
        imputed_data = imputer.fit_transform(scaled_data)
        
        # Inverse transform to get original scale
        imputed_data = scaler.inverse_transform(imputed_data)
        
        # Extract the NO2 values and reshape back
        processed_data = imputed_data[:, 0].reshape(rows, cols)
        
    elif method == 'mice':
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            # Use MICE imputation
            imputer = IterativeImputer(
                max_iter=10,
                random_state=42,
                n_nearest_features=5
            )
            imputed_data = imputer.fit_transform(scaled_data)
            
            # Inverse transform to get original scale
            imputed_data = scaler.inverse_transform(imputed_data)
            
            # Extract the NO2 values and reshape back
            processed_data = imputed_data[:, 0].reshape(rows, cols)
            
        except ImportError:
            print("MICE imputation not available, falling back to KNN")
            return handle_missing_data(data, method='knn')
    
    # Apply floor function to replace negative values with zero
    processed_data = np.maximum(processed_data, 0)
    
    # Apply spatial smoothing to reduce noise
    processed_data = gaussian_filter(processed_data, sigma=0.5)
    
    return processed_data

def forecast_no2_levels(historical_data, periods=30):
    """
    Forecast future NO2 levels using time series analysis.
    
    Parameters:
    - historical_data: Pandas DataFrame with datetime index and NO2 values
    - periods: Number of days to forecast
    
    Returns:
    - forecast: DataFrame with forecasted values and confidence intervals
    - fig: Plotly figure with visualization
    """
    SCALE = 46000  # To convert to µg/m³
    # Make sure data is in the right format
    if isinstance(historical_data, np.ndarray):
        # Convert to dataframe with dates
        dates = [datetime.now() - timedelta(days=i) for i in range(len(historical_data))]
        dates.reverse()
        df = pd.DataFrame({'ds': dates, 'y': np.mean(historical_data, axis=(1, 2))})
    else:
        df = historical_data.copy()
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'ds', 'no2_value': 'y'})
        
    # Ensure no negative values in the historical data
    df['y'] = np.maximum(df['y'], 0)
    
    # Fit Prophet model (handles missing data and seasonality well)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Forecast
    forecast = model.predict(future)
    
    # Apply floor function to ensure no negative forecasted values
    forecast['yhat'] = np.maximum(forecast['yhat'], 0)
    forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
    forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], 0)
    
    # Create visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'] * SCALE,
        mode='markers+lines',
        name='Historical NO2',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'].iloc[-periods:],
        y=forecast['yhat'].iloc[-periods:] * SCALE,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'].iloc[-periods:], forecast['ds'].iloc[-periods:].iloc[::-1]]),
        y=pd.concat([forecast['yhat_upper'].iloc[-periods:], forecast['yhat_lower'].iloc[-periods:].iloc[::-1]]) * SCALE,
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title='NO2 Concentration Forecast (Next 30 Days)',
        xaxis_title='Date',
        yaxis_title='NO2 Concentration (µg/m³)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return forecast, fig

def create_aqi_health_dashboard(no2_value):
    """
    Create a health impact dashboard based on NO2 value.
    
    Parameters:
    - no2_value: NO2 concentration in μg/m³
    
    Returns:
    - fig: Plotly figure with gauge chart
    - health_advice: Health advice based on AQI level
    """
    # Define AQI categories
    categories = [
        {'name': 'Good', 'min': 0, 'max': 40, 'color': 'green', 
         'advice': 'Air quality is good. Enjoy outdoor activities!'},
        {'name': 'Moderate', 'min': 41, 'max': 80, 'color': 'yellow', 
         'advice': 'Unusually sensitive people should consider reducing prolonged outdoor exertion.'},
        {'name': 'Poor', 'min': 81, 'max': 180, 'color': 'orange', 
         'advice': 'People with respiratory issues should limit outdoor exertion.'},
        {'name': 'Very Poor', 'min': 181, 'max': 280, 'color': 'red', 
         'advice': 'Everyone should limit outdoor exertion. Children, elderly, and people with respiratory issues should stay indoors.'},
        {'name': 'Severe', 'min': 281, 'max': 400, 'color': 'purple', 
         'advice': 'Everyone should avoid outdoor activities. Health warnings of emergency conditions for the entire population.'}
    ]
    
    # Determine AQI category
    category = next((cat for cat in categories if cat['min'] <= no2_value <= cat['max']), 
                   categories[-1] if no2_value > categories[-1]['max'] else categories[0])
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = no2_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Air Quality Index (NO2)", 'font': {'size': 24}},
        delta = {'reference': 40, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 400], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': category['color']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.5)'},
                {'range': [41, 80], 'color': 'rgba(255, 255, 0, 0.5)'},
                {'range': [81, 180], 'color': 'rgba(255, 165, 0, 0.5)'},
                {'range': [181, 280], 'color': 'rgba(255, 0, 0, 0.5)'},
                {'range': [281, 400], 'color': 'rgba(128, 0, 128, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 300
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Health advice
    health_advice = {
        'category': category['name'],
        'color': category['color'],
        'advice': category['advice'],
        'cigarette_equivalent': round(no2_value / 22, 1),
        'recommendations': [
            f"Current NO2 level is {no2_value} μg/m³ ({category['name']})",
            f"This is equivalent to smoking {round(no2_value / 22, 1)} cigarettes per day",
            category['advice']
        ]
    }
    
    return fig, health_advice

def compare_locations(data_dict, metric='mean'):
    """
    Compare NO2 levels across different locations.
    
    Parameters:
    - data_dict: Dictionary with location names as keys and NO2 data as values
    - metric: Metric to use for comparison ('mean', 'max', 'min', 'std')
    
    Returns:
    - fig: Plotly figure with bar chart comparison
    """
    locations = []
    values = []
    
    for location, data in data_dict.items():
        # Apply floor function to ensure no negative values
        data_no_negatives = np.maximum(data, 0)
        
        locations.append(location)
        if metric == 'mean':
            values.append(np.nanmean(data_no_negatives) * 46000)  # Convert to μg/m³
        elif metric == 'max':
            values.append(np.nanmax(data_no_negatives) * 46000)
        elif metric == 'min':
            values.append(np.nanmin(data_no_negatives) * 46000)
        elif metric == 'std':
            values.append(np.nanstd(data_no_negatives) * 46000)
    
    # Determine color based on values (lower is better for NO2)
    colors = ['green' if v <= 40 else 'yellow' if v <= 80 else 'orange' if v <= 180 else 'red' if v <= 280 else 'purple' for v in values]
    
    fig = go.Figure(go.Bar(
        x=locations,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f} μg/m³" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'NO2 Comparison Across Locations ({metric.capitalize()})',
        xaxis_title='Location',
        yaxis_title='NO2 Concentration (μg/m³)',
        height=500
    )
    
    return fig

def explain_no2_trend(forecast_df, periods=30):
    """
    Generate a human-readable reason for the forecasted NO2 trend.
    """
    # Get the forecasted values for the period
    yhat = forecast_df['yhat'].iloc[-periods:]
    dates = forecast_df['ds'].iloc[-periods:]
    start_val = yhat.iloc[0]
    end_val = yhat.iloc[-1]
    percent_change = ((end_val - start_val) / (start_val + 1e-6)) * 100
    # Get the main months in the forecast
    months = dates.dt.month.unique()
    month_names = [datetime(2000, m, 1).strftime('%B') for m in months]
    # Default reason
    reason = "NO₂ levels are expected to remain stable."
    # Rising trend
    if percent_change > 5:
        if 5 in months or 6 in months:  # May/June
            reason = ("NO₂ levels are expected to rise in the coming weeks, likely due to increased vehicle and industrial activity during the summer months, "
                      "and less rainfall to clean the air.")
        elif 10 in months or 11 in months:  # Oct/Nov
            reason = ("A rise in NO₂ is forecasted, possibly due to post-monsoon crop burning, Diwali fireworks, and increased heating needs in winter.")
        else:
            reason = ("NO₂ levels are forecasted to rise, which may be due to increased emissions from vehicles, industry, or less favorable weather conditions for pollution dispersion.")
    # Falling trend
    elif percent_change < -5:
        if 7 in months or 8 in months:  # July/Aug
            reason = ("NO₂ levels are expected to drop, likely because of the onset of the monsoon season, which helps wash pollutants out of the atmosphere.")
        elif 3 in months or 4 in months:  # Mar/Apr
            reason = ("A decrease in NO₂ is forecasted, possibly due to spring winds and improved weather conditions that help disperse pollution.")
        else:
            reason = ("NO₂ levels are forecasted to decrease, likely due to improved weather conditions or reduced emissions.")
    # Stable trend
    else:
        reason = ("NO₂ levels are expected to remain relatively stable, indicating no major changes in weather or emission sources during this period.")
    return reason