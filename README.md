# gdg-hackathon - Enhanced NO2 Air Quality Analysis Dashboard
Downscaling of Satellite based air quality map using AI/ML

## 🌟 Enhanced Features 
Our solution offers a comprehensive approach to NO2 air quality analysis with these advanced capabilities:

1. **AI-Powered Downscaling**: Uses an ensemble model combining RandomForest, XGBoost, and Neural Networks to generate high-resolution NO2 maps from coarse satellite data.

2. **Time Series Forecasting**: Predicts future NO2 levels for the next 30 days, helping plan outdoor activities and inform policy decisions.

3. **Multi-City Comparison**: Compare air quality across different cities with interactive maps and visualizations.

4. **Health Impact Analysis**: Translates NO2 levels into health impacts and provides practical advice for different air quality conditions.

5. **Seasonal Analysis**: Identifies patterns in air quality across different seasons to understand when pollution is at its worst.

6. **Location-Specific Recommendations**: Generates tailored recommendations for reducing NO2 pollution based on city-specific characteristics.

7. **Correlation Analysis**: Identifies the relationship between NO2 levels and factors like population density, traffic, and green cover.

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/gdg-hackathon.git
cd gdg-hackathon

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## 📊 Key Components

- **Model**: Ensemble approach combining RandomForest, XGBoost, and Neural Networks for better downscaling accuracy
- **Interactive Dashboard**: Streamlit-based interface with multiple tabs for different analysis types
- **Forecasting Engine**: Prophet-based time series forecasting for future NO2 predictions
- **Geospatial Analysis**: Integration with Folium for interactive mapping capabilities
- **Health Impact Assessment**: Conversion of NO2 levels to health metrics like cigarette equivalents

## 📚 Original Description

Develop an AI/ML (Artificial Intelligence/Machine Learning) model to generate fine spatial resolution air quality map from coarse resolution satellite data. It should utilise existing python-based ML libraries. Developed model need to be validated with unseen independent data. 

Challenge: 
- To utilise large satellite data having gaps under cloudy conditions 
- To select suitable ML algorithm and ensure optimal fitting of ML model for desired accuracy 
- To validate model output with unseen independent data 

Usage: To enhance air quality knowledge, Sharpen focus at local level 

Users: Researchers and government bodies monitoring/working on air quality assessment 

Available Solutions (if Yes, reasons for not using them): Individual components are available, comprehensive and proven solution does not exist. 

Desired Outcome: Fine resolution air quality map of NO2

## 📁 Data Source

1. Satellite derived daily Tropospheric NO2 from available sources:
   - Daily Tropospheric NO2 from TROPOMI/Sentinel-5p
   - Daily Tropospheric NO2 from OMI/Aura
   
2. Ground-based NO2 concentration data from CPCB

## 📊 Machine Learning Algorithms
Our enhanced solution uses an ensemble approach combining:
- RandomForest
- XGBoost
- Neural Networks

## 🧪 Enhanced Features in Detail

### AI-Powered Downscaling
Our model goes beyond traditional approaches:
- Uses sophisticated feature engineering with sinusoidal transformations
- Implements ensemble learning for more robust predictions
- Provides uncertainty estimates with confidence intervals

### Time Series Forecasting
- Prophet-based forecasting that accounts for seasonality
- Forecasts up to 90 days in advance
- Provides confidence intervals to account for uncertainty

### Health Impact Analysis
- Converts NO2 levels to cigarette equivalents for intuitive understanding
- Provides health recommendations based on AQI categories
- Uses color-coded indicators for quick risk assessment

### Multi-City Comparison
- Interactive map interface to visualize NO2 levels across cities
- Correlation analysis with socio-economic and environmental factors
- City-specific recommendations based on analysis results

Data source: 
1. Satellite derived daily Tropospheric NO2 from either of the following links: (a) Daily Tropospheric NO2 from TROPOMI/Sentinel-5p – Swath data 
https://search.earthdata.nasa.gov/search/granules?p=C2089270961-
GES_DISC&pg[0][v]=f&pg[0][gsk]=-start_date&q=tropomi%20no2&tl=1726635700.002!3!!  
(b)	Daily Tropospheric NO2 from TROPOMI/Sentinel-5p (using google earth engine) – gridded geotif format  
https://developers.google.com/earth-
engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2#description  
(c)	Daily Tropospheric NO2 from OMI/Aura – gridded data 
https://search.earthdata.nasa.gov/search/granules?p=C1266136111GES_DISC&pg[0][v]=f&pg[0][gsk]=start_date&q=omi%20tropospheric%20no2&tl=1726635700.002!3!! 
(d)	Daily Tropospheric NO2 from OMI/Aura – gridded data https://measures.gesdisc.eosdis.nasa.gov/data/MINDS/OMI_MINDS_NO2d.1.1/2024/ 
 
2. Either of the above data (daily tropospheric NO2) to be used in conjunction with ground-based NO2 concentration monitored by CPCB: https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqmlanding (go to Advance Search to download data for different stations) 
 
Machine learning algorithm 
Generally, Random Forest, XGBoost and Neural Network (ANN/CNN) are good for downscaling. However, students may explore different AI/ML algorithms and can decide themselves which algorithm to be used.  
 
