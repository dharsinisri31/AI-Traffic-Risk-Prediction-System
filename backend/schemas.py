from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class HealthResponse(BaseModel):
    status: str = "healthy"

class CongestionPredictionResponse(BaseModel):
    congestion_level: str
    confidence: Optional[float] = None
    recommendation: str

class AccidentPredictionResponse(BaseModel):
    risk_level: str
    risk_percentage: Optional[float] = None
    recommendation: str
    disclaimer: str = "This prediction is an AI-based accident risk estimation for decision-support and does not guarantee accident occurrence."

class CongestionInputSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    temp: float = Field(default=288.15, description="Feature: temp")
    rain_1h: float = Field(default=0.0, description="Feature: rain_1h")
    snow_1h: float = Field(default=0.0, description="Feature: snow_1h")
    clouds_all: int = Field(default=0, description="Feature: clouds_all")
    hour: int = Field(default=12, description="Feature: hour")
    day: int = Field(default=1, description="Feature: day")
    month: int = Field(default=1, description="Feature: month")
    year: int = Field(default=2024, description="Feature: year")
    day_of_week: int = Field(default=0, description="Feature: day_of_week")
    is_weekend: int = Field(default=0, description="Feature: is_weekend")
    is_peak_hour: int = Field(default=0, description="Feature: is_peak_hour")
    traffic_risk: float = Field(default=0.0, description="Feature: traffic_risk")
    rain_risk: float = Field(default=0.0, description="Feature: rain_risk")
    snow_risk: float = Field(default=0.0, description="Feature: snow_risk")
    weather_risk: float = Field(default=0.0, description="Feature: weather_risk")
    holiday_Columbus_Day: int = Field(default=0, alias="holiday_Columbus Day", description="Feature: holiday_Columbus Day")
    holiday_Independence_Day: int = Field(default=0, alias="holiday_Independence Day", description="Feature: holiday_Independence Day")
    holiday_Labor_Day: int = Field(default=0, alias="holiday_Labor Day", description="Feature: holiday_Labor Day")
    holiday_Martin_Luther_King_Jr_Day: int = Field(default=0, alias="holiday_Martin Luther King Jr Day", description="Feature: holiday_Martin Luther King Jr Day")
    holiday_Memorial_Day: int = Field(default=0, alias="holiday_Memorial Day", description="Feature: holiday_Memorial Day")
    holiday_New_Years_Day: int = Field(default=0, alias="holiday_New Years Day", description="Feature: holiday_New Years Day")
    holiday_None: int = Field(default=1, description="Feature: holiday_None")
    holiday_State_Fair: int = Field(default=0, alias="holiday_State Fair", description="Feature: holiday_State Fair")
    holiday_Thanksgiving_Day: int = Field(default=0, alias="holiday_Thanksgiving Day", description="Feature: holiday_Thanksgiving Day")
    holiday_Veterans_Day: int = Field(default=0, alias="holiday_Veterans Day", description="Feature: holiday_Veterans Day")
    holiday_Washingtons_Birthday: int = Field(default=0, alias="holiday_Washingtons Birthday", description="Feature: holiday_Washingtons Birthday")
    weather_main_Clouds: int = Field(default=0, description="Feature: weather_main_Clouds")
    weather_main_Drizzle: int = Field(default=0, description="Feature: weather_main_Drizzle")
    weather_main_Fog: int = Field(default=0, description="Feature: weather_main_Fog")
    weather_main_Haze: int = Field(default=0, description="Feature: weather_main_Haze")
    weather_main_Mist: int = Field(default=0, description="Feature: weather_main_Mist")
    weather_main_Rain: int = Field(default=0, description="Feature: weather_main_Rain")
    weather_main_Smoke: int = Field(default=0, description="Feature: weather_main_Smoke")
    weather_main_Snow: int = Field(default=0, description="Feature: weather_main_Snow")
    weather_main_Squall: int = Field(default=0, description="Feature: weather_main_Squall")
    weather_main_Thunderstorm: int = Field(default=0, description="Feature: weather_main_Thunderstorm")
    weather_description_Sky_is_Clear: int = Field(default=0, alias="weather_description_Sky is Clear", description="Feature: weather_description_Sky is Clear")
    weather_description_broken_clouds: int = Field(default=0, alias="weather_description_broken clouds", description="Feature: weather_description_broken clouds")
    weather_description_drizzle: int = Field(default=0, description="Feature: weather_description_drizzle")
    weather_description_few_clouds: int = Field(default=0, alias="weather_description_few clouds", description="Feature: weather_description_few clouds")
    weather_description_fog: int = Field(default=0, description="Feature: weather_description_fog")
    weather_description_freezing_rain: int = Field(default=0, alias="weather_description_freezing rain", description="Feature: weather_description_freezing rain")
    weather_description_haze: int = Field(default=0, description="Feature: weather_description_haze")
    weather_description_heavy_intensity_drizzle: int = Field(default=0, alias="weather_description_heavy intensity drizzle", description="Feature: weather_description_heavy intensity drizzle")
    weather_description_heavy_intensity_rain: int = Field(default=0, alias="weather_description_heavy intensity rain", description="Feature: weather_description_heavy intensity rain")
    weather_description_heavy_snow: int = Field(default=0, alias="weather_description_heavy snow", description="Feature: weather_description_heavy snow")
    weather_description_light_intensity_drizzle: int = Field(default=0, alias="weather_description_light intensity drizzle", description="Feature: weather_description_light intensity drizzle")
    weather_description_light_intensity_shower_rain: int = Field(default=0, alias="weather_description_light intensity shower rain", description="Feature: weather_description_light intensity shower rain")
    weather_description_light_rain: int = Field(default=0, alias="weather_description_light rain", description="Feature: weather_description_light rain")
    weather_description_light_rain_and_snow: int = Field(default=0, alias="weather_description_light rain and snow", description="Feature: weather_description_light rain and snow")
    weather_description_light_shower_snow: int = Field(default=0, alias="weather_description_light shower snow", description="Feature: weather_description_light shower snow")
    weather_description_light_snow: int = Field(default=0, alias="weather_description_light snow", description="Feature: weather_description_light snow")
    weather_description_mist: int = Field(default=0, description="Feature: weather_description_mist")
    weather_description_moderate_rain: int = Field(default=0, alias="weather_description_moderate rain", description="Feature: weather_description_moderate rain")
    weather_description_overcast_clouds: int = Field(default=0, alias="weather_description_overcast clouds", description="Feature: weather_description_overcast clouds")
    weather_description_proximity_shower_rain: int = Field(default=0, alias="weather_description_proximity shower rain", description="Feature: weather_description_proximity shower rain")
    weather_description_proximity_thunderstorm: int = Field(default=0, alias="weather_description_proximity thunderstorm", description="Feature: weather_description_proximity thunderstorm")
    weather_description_proximity_thunderstorm_with_drizzle: int = Field(default=0, alias="weather_description_proximity thunderstorm with drizzle", description="Feature: weather_description_proximity thunderstorm with drizzle")
    weather_description_proximity_thunderstorm_with_rain: int = Field(default=0, alias="weather_description_proximity thunderstorm with rain", description="Feature: weather_description_proximity thunderstorm with rain")
    weather_description_scattered_clouds: int = Field(default=0, alias="weather_description_scattered clouds", description="Feature: weather_description_scattered clouds")
    weather_description_shower_drizzle: int = Field(default=0, alias="weather_description_shower drizzle", description="Feature: weather_description_shower drizzle")
    weather_description_shower_snow: int = Field(default=0, alias="weather_description_shower snow", description="Feature: weather_description_shower snow")
    weather_description_sky_is_clear: int = Field(default=0, alias="weather_description_sky is clear", description="Feature: weather_description_sky is clear")
    weather_description_sleet: int = Field(default=0, description="Feature: weather_description_sleet")
    weather_description_smoke: int = Field(default=0, description="Feature: weather_description_smoke")
    weather_description_snow: int = Field(default=0, description="Feature: weather_description_snow")
    weather_description_thunderstorm: int = Field(default=0, description="Feature: weather_description_thunderstorm")
    weather_description_thunderstorm_with_drizzle: int = Field(default=0, alias="weather_description_thunderstorm with drizzle", description="Feature: weather_description_thunderstorm with drizzle")
    weather_description_thunderstorm_with_heavy_rain: int = Field(default=0, alias="weather_description_thunderstorm with heavy rain", description="Feature: weather_description_thunderstorm with heavy rain")
    weather_description_thunderstorm_with_light_drizzle: int = Field(default=0, alias="weather_description_thunderstorm with light drizzle", description="Feature: weather_description_thunderstorm with light drizzle")
    weather_description_thunderstorm_with_light_rain: int = Field(default=0, alias="weather_description_thunderstorm with light rain", description="Feature: weather_description_thunderstorm with light rain")
    weather_description_thunderstorm_with_rain: int = Field(default=0, alias="weather_description_thunderstorm with rain", description="Feature: weather_description_thunderstorm with rain")
    weather_description_very_heavy_rain: int = Field(default=0, alias="weather_description_very heavy rain", description="Feature: weather_description_very heavy rain")

# AccidentInputSchema shares the exact 73 feature columns
class AccidentInputSchema(CongestionInputSchema):
    pass
