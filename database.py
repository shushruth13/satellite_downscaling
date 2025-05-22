import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Use SQLite for development
DATABASE_URL = "sqlite:///./air_quality.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class SatelliteData(Base):
    __tablename__ = "satellite_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    no2_value = Column(Float)
    resolution = Column(Float)  # in kilometers
    source = Column(String)
    season = Column(String)  # Winter, Summer, Monsoon, Post-Monsoon
    measurements = relationship("GroundMeasurement", back_populates="satellite_data")
    forecasts = relationship("NO2Forecast", back_populates="satellite_data")

class GroundMeasurement(Base):
    __tablename__ = "ground_measurements"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    no2_value = Column(Float)
    station_name = Column(String)
    station_type = Column(String)  # urban, suburban, rural, traffic, background
    satellite_data_id = Column(Integer, ForeignKey("satellite_data.id"))
    satellite_data = relationship("SatelliteData", back_populates="measurements")

class NO2Forecast(Base):
    __tablename__ = "no2_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    creation_timestamp = Column(DateTime, default=datetime.utcnow)
    forecast_date = Column(DateTime)
    latitude = Column(Float)
    longitude = Column(Float)
    predicted_no2 = Column(Float)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    model_version = Column(String)
    satellite_data_id = Column(Integer, ForeignKey("satellite_data.id"))
    satellite_data = relationship("SatelliteData", back_populates="forecasts")

class HealthImpact(Base):
    __tablename__ = "health_impacts"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    no2_value = Column(Float)
    aqi_category = Column(String)  # Good, Moderate, Poor, Very Poor, Severe
    cigarette_equivalent = Column(Float)
    health_advice = Column(Text)
    
class Location(Base):
    __tablename__ = "locations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    latitude = Column(Float)
    longitude = Column(Float)
    population = Column(Integer, nullable=True)
    area_type = Column(String)  # urban, suburban, rural
    has_industry = Column(Boolean, default=False)
    traffic_density = Column(String, nullable=True)  # low, medium, high
    green_cover_percentage = Column(Float, nullable=True)
    
class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    model_version = Column(String)
    rmse = Column(Float)
    r2 = Column(Float)
    mse = Column(Float)
    hyperparameters = Column(JSON, nullable=True)
    training_time = Column(Float, nullable=True)  # seconds
    description = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
