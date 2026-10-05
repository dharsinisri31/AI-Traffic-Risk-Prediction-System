"""
AI Traffic Risk Prediction System - Streamlit Frontend Application (Dark Theme)

Architecture:
- Dark theme styling with large, presentation-ready typography and high contrast
- Multi-page navigation in the sidebar (Congestion Predictor, Accident Risk Predictor, Dual Predictor, Analytics Board)
- Streamlined input fields with manual risk index and date sliders removed for simplicity
- Real-time API communication with FastAPI backend
"""

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK THEME WITH LARGER TYPOGRAPHY
# ==============================================================================
st.set_page_config(
    page_title="AI Traffic Risk Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Custom Sleek Dark Theme CSS with enlarged font sizes
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 17px;
    }

    /* Main Canvas Background */
    .stApp {
        background: radial-gradient(circle at top right, #0F172A 0%, #090D16 60%, #04070E 100%);
        color: #F8FAFC;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Sidebar Text Size */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        font-size: 1.05rem !important;
    }

    /* Main Page Titles */
    h1 {
        color: #F8FAFC !important;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.3rem !important;
    }
    h2 {
        color: #F8FAFC !important;
        font-size: 2.0rem !important;
        font-weight: 700 !important;
    }
    h3 {
        color: #F8FAFC !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    /* Input Labels - Enlarged & High Contrast */
    label[data-testid="stWidgetLabel"] p {
        font-size: 1.18rem !important;
        font-weight: 600 !important;
        color: #F1F5F9 !important;
        margin-bottom: 6px !important;
    }

    /* Input Controls Text Size */
    div[data-baseweb="input"] input, div[data-baseweb="select"] {
        font-size: 1.12rem !important;
        border-radius: 8px !important;
    }

    /* Status Badges */
    .badge-online {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.0rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .badge-offline {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.0rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    /* Output Risk Cards */
    .result-box-low {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        color: #FFFFFF;
        box-shadow: 0 4px 24px rgba(16, 185, 129, 0.25);
        margin-top: 15px;
    }
    .result-box-medium {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #F59E0B;
        border-radius: 14px;
        padding: 24px;
        color: #FFFFFF;
        box-shadow: 0 4px 24px rgba(245, 158, 11, 0.25);
        margin-top: 15px;
    }
    .result-box-high {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        color: #FFFFFF;
        box-shadow: 0 4px 24px rgba(239, 68, 68, 0.25);
        margin-top: 15px;
    }

    .disclaimer-tag {
        font-size: 0.95rem;
        color: #94A3B8;
        border-left: 4px solid #F59E0B;
        padding-left: 12px;
        margin-top: 14px;
        background: rgba(245, 158, 11, 0.1);
        padding: 10px 14px;
        border-radius: 6px;
    }

    /* Enlarged Action Buttons */
    div.stButton > button:first-child {
        border-radius: 12px;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        transition: all 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. HELPER FUNCTIONS: API COMMUNICATION
# ==============================================================================
def check_backend_health() -> tuple[bool, str]:
    """Pings FastAPI GET /health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("models_loaded", False):
                return True, "FastAPI Backend Online & Models Loaded"
            return True, "FastAPI Backend Online (Models Not Loaded)"
        return False, f"Backend returned HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to FastAPI backend"
    except Exception as e:
        return False, str(e)


def call_prediction_api(endpoint: str, payload: dict) -> tuple[bool, dict | str]:
    """Dispatches POST request to prediction endpoints."""
    url = f"{API_URL}/{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=6)
        if response.status_code == 200:
            return True, response.json()
        try:
            err_data = response.json()
            err_msg = err_data.get("detail", f"Server error {response.status_code}")
        except Exception:
            err_msg = response.text or f"HTTP {response.status_code}"
        return False, err_msg
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to FastAPI server at '{API_URL}'."
    except Exception as e:
        return False, f"Request failed: {str(e)}"


# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("## 🚦 AI Traffic System")
    
    # Rounded Page Navigation in Sidebar
    page = st.radio(
        "Navigation",
        options=[
            "🚦 Traffic Congestion Predictor",
            "⚠️ Accident Risk Predictor",
            "⚡ Dual AI Predictor",
            "📊 Traffic Analytics Board",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("### 🔌 API Status")
    is_online, health_text = check_backend_health()
    if is_online:
        st.markdown('<span class="badge-online">🟢 Online</span>', unsafe_allow_html=True)
        st.caption(f"<span style='font-size:0.95rem;'>{health_text}</span>", unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">🔴 Offline</span>', unsafe_allow_html=True)
        st.caption(f"<span style='font-size:0.95rem;'>{health_text}</span>", unsafe_allow_html=True)
        st.info("💡 Start backend: `python -m uvicorn backend.main:app --reload`")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Scenarios")
    preset = st.selectbox(
        "Load Preset Scenario",
        options=[
            "Custom Inputs",
            "Peak Rush Hour Traffic",
            "Clear Afternoon (Low Risk)",
            "Heavy Rainstorm & Night (High Risk)",
        ],
        index=0,
    )

# Preset Configuration
if preset == "Peak Rush Hour Traffic":
    p_hour, p_temp_c, p_rain, p_snow, p_clouds = 18, 20.0, 0.2, 0.0, 60
    p_dow_idx, p_weekend, p_peak = 0, 0, 1
    p_holiday, p_w_main, p_w_desc = "None", "Clouds", "broken clouds"

elif preset == "Clear Afternoon (Low Risk)":
    p_hour, p_temp_c, p_rain, p_snow, p_clouds = 14, 24.0, 0.0, 0.0, 10
    p_dow_idx, p_weekend, p_peak = 2, 0, 0
    p_holiday, p_w_main, p_w_desc = "None", "Clear", "sky is clear"

elif preset == "Heavy Rainstorm & Night (High Risk)":
    p_hour, p_temp_c, p_rain, p_snow, p_clouds = 22, 12.0, 18.0, 0.0, 100
    p_dow_idx, p_weekend, p_peak = 4, 0, 0
    p_holiday, p_w_main, p_w_desc = "None", "Rain", "heavy intensity rain"

else:
    p_hour, p_temp_c, p_rain, p_snow, p_clouds = 12, 18.0, 0.0, 0.0, 40
    p_dow_idx, p_weekend, p_peak = 0, 0, 0
    p_holiday, p_w_main, p_w_desc = "None", "Clouds", "scattered clouds"


# ==============================================================================
# 4. REUSABLE CLEAN INPUT FORM (ESSENTIAL PARAMETERS ONLY)
# ==============================================================================
def render_clean_inputs() -> dict:
    """Renders direct, essential feature inputs in a clean 2-column layout."""
    col1, col2 = st.columns(2)

    with col1:
        hour = st.slider("Hour of Day (0 - 23)", min_value=0, max_value=23, value=p_hour)
        
        dow_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_dow = st.selectbox("Day of Week", options=dow_list, index=p_dow_idx)
        dow_val = dow_list.index(selected_dow)

        is_weekend = st.selectbox(
            "Is Weekend?",
            options=[0, 1],
            index=p_weekend,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )

        is_peak_hour = st.selectbox(
            "Is Peak Rush Hour?",
            options=[0, 1],
            index=p_peak,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )

        holidays = [
            "None", "Columbus Day", "Independence Day", "Labor Day",
            "Martin Luther King Jr Day", "Memorial Day", "New Years Day",
            "State Fair", "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"
        ]
        holiday_idx = holidays.index(p_holiday) if p_holiday in holidays else 0
        holiday = st.selectbox("Holiday", options=holidays, index=holiday_idx)

    with col2:
        temp_c = st.number_input("Temperature (°C)", min_value=-30.0, max_value=55.0, value=float(p_temp_c), step=0.5)
        temp_k = round(temp_c + 273.15, 2)
        st.caption(f"<span style='font-size:1.0rem; color:#94A3B8;'>ℹ️ Temperature in Kelvin: <b>{temp_k} K</b></span>", unsafe_allow_html=True)

        rain_1h = st.number_input("Rainfall (mm)", min_value=0.0, max_value=150.0, value=float(p_rain), step=0.1)
        snow_1h = st.number_input("Snowfall (mm)", min_value=0.0, max_value=100.0, value=float(p_snow), step=0.1)
        clouds_all = st.slider("Cloud Coverage (%)", min_value=0, max_value=100, value=int(p_clouds))

        w_main_opts = ["Clouds", "Clear", "Rain", "Drizzle", "Mist", "Fog", "Haze", "Snow", "Thunderstorm", "Smoke", "Squall"]
        w_main_idx = w_main_opts.index(p_w_main) if p_w_main in w_main_opts else 0
        weather_main = st.selectbox("Weather Category", options=w_main_opts, index=w_main_idx)

        w_desc_opts = [
            "sky is clear", "broken clouds", "scattered clouds", "few clouds", "overcast clouds",
            "light rain", "moderate rain", "heavy intensity rain", "very heavy rain",
            "light intensity drizzle", "drizzle", "heavy intensity drizzle", "freezing rain",
            "light snow", "snow", "heavy snow", "sleet", "shower snow", "light shower snow",
            "mist", "fog", "haze", "smoke", "thunderstorm",
            "thunderstorm with light rain", "thunderstorm with rain", "thunderstorm with heavy rain",
            "proximity shower rain", "proximity thunderstorm"
        ]
        w_desc_idx = w_desc_opts.index(p_w_desc) if p_w_desc in w_desc_opts else 0
        weather_description = st.selectbox("Weather Condition", options=w_desc_opts, index=w_desc_idx)

    # Automated background risk indices calculations
    auto_rain_risk = round(min(1.0, rain_1h / 15.0), 2)
    auto_snow_risk = round(min(1.0, snow_1h / 10.0), 2)
    auto_weather_risk = round(max(auto_rain_risk, auto_snow_risk, 0.1 if clouds_all > 70 else 0.0), 2)
    auto_traffic_risk = 0.85 if is_peak_hour == 1 else (0.5 if (hour >= 8 and hour <= 20) else 0.15)

    return {
        "temp": temp_k,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "hour": hour,
        "day": 15,
        "month": 10,
        "year": 2024,
        "day_of_week": dow_val,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "traffic_risk": auto_traffic_risk,
        "rain_risk": auto_rain_risk,
        "snow_risk": auto_snow_risk,
        "weather_risk": auto_weather_risk,
        "holiday": holiday,
        "weather_main": weather_main,
        "weather_description": weather_description,
    }


# ==============================================================================
# 5. PAGE: TRAFFIC CONGESTION PREDICTOR
# ==============================================================================
if page == "🚦 Traffic Congestion Predictor":
    st.title("🚦 Traffic Congestion Predictor")
    st.markdown("<p style='font-size:1.2rem; color:#94A3B8;'>Enter environmental & traffic values to predict Congestion Class (<b>Low</b>, <b>Medium</b>, or <b>High</b>).</p>", unsafe_allow_html=True)

    payload = render_clean_inputs()

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Congestion Level", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("Calculating Traffic Congestion..."):
            ok, result = call_prediction_api("predict/congestion", payload)
            if ok:
                level = result.get("congestion_level", "Unknown")
                conf = result.get("confidence")
                rec = result.get("recommendation", "")

                card_style = "result-box-low" if level == "Low" else ("result-box-medium" if level == "Medium" else "result-box-high")
                icon = "🟢" if level == "Low" else ("🟡" if level == "Medium" else "🔴")

                st.markdown(
                    f"""
                    <div class="{card_style}">
                        <div style="font-size:1.0rem; letter-spacing:1px; font-weight:700; opacity:0.85;">PREDICTED CONGESTION CLASS</div>
                        <h1 style="margin:8px 0 12px 0; color:#FFFFFF !important; font-size:2.8rem !important;">{icon} {level} Congestion</h1>
                        <p style="margin:0 0 14px 0; font-size:1.25rem;"><b>Model Confidence Score:</b> {f'{conf:.1%}' if conf is not None else 'N/A'}</p>
                        <div style="background:rgba(0,0,0,0.4); padding:14px 18px; border-radius:10px; border:1px solid rgba(255,255,255,0.15); font-size:1.15rem;">
                            💡 <b>Traffic Advisory:</b> {rec}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"❌ Prediction Failed: {result}")


# ==============================================================================
# 6. PAGE: ACCIDENT RISK PREDICTOR
# ==============================================================================
elif page == "⚠️ Accident Risk Predictor":
    st.title("⚠️ Estimated Accident Risk Predictor")
    st.markdown("<p style='font-size:1.2rem; color:#94A3B8;'>Enter environmental & road parameters to estimate Accident Risk (<b>Low</b>, <b>Moderate</b>, or <b>High</b>).</p>", unsafe_allow_html=True)

    payload = render_clean_inputs()

    st.markdown("---")
    predict_btn = st.button("⚠️ Estimate Accident Risk", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("Estimating Accident Risk Probability..."):
            ok, result = call_prediction_api("predict/accident", payload)
            if ok:
                risk = result.get("risk_level", "Unknown")
                pct = result.get("risk_percentage")
                rec = result.get("recommendation", "")
                disclaimer = result.get("disclaimer", "")

                card_style = "result-box-low" if risk == "Low" else ("result-box-medium" if risk == "Moderate" else "result-box-high")
                icon = "🟢" if risk == "Low" else ("🟠" if risk == "Moderate" else "🔴")

                st.markdown(
                    f"""
                    <div class="{card_style}">
                        <div style="font-size:1.0rem; letter-spacing:1px; font-weight:700; opacity:0.85;">ESTIMATED ACCIDENT RISK CLASS</div>
                        <h1 style="margin:8px 0 12px 0; color:#FFFFFF !important; font-size:2.8rem !important;">{icon} {risk} Risk</h1>
                        <p style="margin:0 0 14px 0; font-size:1.25rem;"><b>Estimated Risk Probability:</b> {f'{pct:.1f}%' if pct is not None else 'N/A'}</p>
                        <div style="background:rgba(0,0,0,0.4); padding:14px 18px; border-radius:10px; border:1px solid rgba(255,255,255,0.15); font-size:1.15rem;">
                            🛡️ <b>Safety Advisory:</b> {rec}
                        </div>
                        <div class="disclaimer-tag">
                            ⚠️ <b>Disclaimer:</b> {disclaimer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"❌ Prediction Failed: {result}")


# ==============================================================================
# 7. PAGE: DUAL PREDICTOR (BOTH PREDICTIONS TOGETHER)
# ==============================================================================
elif page == "⚡ Dual AI Predictor":
    st.title("⚡ Unified AI Traffic & Accident Risk Predictor")
    st.markdown("<p style='font-size:1.2rem; color:#94A3B8;'>Simultaneously computes Traffic Congestion and Estimated Accident Risk.</p>", unsafe_allow_html=True)

    payload = render_clean_inputs()

    st.markdown("---")
    predict_both_btn = st.button("⚡ Run Dual AI Predictions", use_container_width=True, type="primary")

    if predict_both_btn:
        col_c, col_a = st.columns(2)

        with col_c:
            with st.spinner("Predicting Congestion..."):
                ok_c, res_c = call_prediction_api("predict/congestion", payload)
                if ok_c:
                    level = res_c.get("congestion_level", "Unknown")
                    conf = res_c.get("confidence")
                    rec = res_c.get("recommendation", "")
                    card_style = "result-box-low" if level == "Low" else ("result-box-medium" if level == "Medium" else "result-box-high")
                    icon = "🟢" if level == "Low" else ("🟡" if level == "Medium" else "🔴")

                    st.markdown(
                        f"""
                        <div class="{card_style}">
                            <div style="font-size:0.95rem; font-weight:700; opacity:0.85;">TRAFFIC CONGESTION</div>
                            <h2 style="margin:8px 0 10px 0; color:#FFFFFF !important; font-size:2.2rem !important;">{icon} {level}</h2>
                            <p style="margin:0 0 12px 0; font-size:1.15rem;"><b>Confidence:</b> {f'{conf:.1%}' if conf is not None else 'N/A'}</p>
                            <div style="background:rgba(0,0,0,0.4); padding:12px 16px; border-radius:8px; font-size:1.05rem;">
                                💡 {rec}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Congestion Error: {res_c}")

        with col_a:
            with st.spinner("Estimating Accident Risk..."):
                ok_a, res_a = call_prediction_api("predict/accident", payload)
                if ok_a:
                    risk = res_a.get("risk_level", "Unknown")
                    pct = res_a.get("risk_percentage")
                    rec = res_a.get("recommendation", "")
                    disclaimer = res_a.get("disclaimer", "")
                    card_style = "result-box-low" if risk == "Low" else ("result-box-medium" if risk == "Moderate" else "result-box-high")
                    icon = "🟢" if risk == "Low" else ("🟠" if risk == "Moderate" else "🔴")

                    st.markdown(
                        f"""
                        <div class="{card_style}">
                            <div style="font-size:0.95rem; font-weight:700; opacity:0.85;">ESTIMATED ACCIDENT RISK</div>
                            <h2 style="margin:8px 0 10px 0; color:#FFFFFF !important; font-size:2.2rem !important;">{icon} {risk}</h2>
                            <p style="margin:0 0 12px 0; font-size:1.15rem;"><b>Risk Probability:</b> {f'{pct:.1f}%' if pct is not None else 'N/A'}</p>
                            <div style="background:rgba(0,0,0,0.4); padding:12px 16px; border-radius:8px; font-size:1.05rem;">
                                🛡️ {rec}
                            </div>
                            <div class="disclaimer-tag">
                                ⚠️ {disclaimer}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Accident Risk Error: {res_a}")


# ==============================================================================
# 8. PAGE: ANALYTICS BOARD
# ==============================================================================
elif page == "📊 Traffic Analytics Board":
    st.title("📊 Traffic Analytics & Risk Insights")
    st.markdown("<p style='font-size:1.2rem; color:#94A3B8;'>Visualizing 24-hour cycle patterns, weather correlations, and risk distributions.</p>", unsafe_allow_html=True)

    # 24-Hour Cycle Trend
    hours = list(range(24))
    congestion_curve = [
        0.05, 0.03, 0.02, 0.02, 0.04, 0.15, 0.45, 0.85, 0.92, 0.70, 0.55, 0.58,
        0.62, 0.60, 0.58, 0.65, 0.82, 0.95, 0.90, 0.72, 0.50, 0.35, 0.20, 0.10
    ]
    accident_curve = [
        0.12, 0.15, 0.18, 0.14, 0.10, 0.25, 0.50, 0.78, 0.85, 0.62, 0.48, 0.50,
        0.55, 0.52, 0.50, 0.60, 0.75, 0.88, 0.82, 0.65, 0.55, 0.45, 0.35, 0.22
    ]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=hours, y=congestion_curve,
        mode="lines+markers", name="Congestion Probability",
        line=dict(color="#3B82F6", width=3)
    ))
    fig_trend.add_trace(go.Scatter(
        x=hours, y=accident_curve,
        mode="lines+markers", name="Estimated Accident Risk",
        line=dict(color="#EF4444", width=3, dash="dash")
    ))
    fig_trend.update_layout(
        title="24-Hour Congestion vs Accident Risk Profile",
        xaxis_title="Hour of Day (0 - 23)",
        yaxis_title="Normalized Index (0.0 - 1.0)",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=420,
        hovermode="x unified",
        font=dict(size=14),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        weather_categories = ["Clear", "Clouds", "Drizzle", "Rain", "Thunderstorm", "Snow", "Fog"]
        weather_risk_vals = [0.15, 0.30, 0.48, 0.75, 0.92, 0.88, 0.80]
        fig_bar = px.bar(
            x=weather_categories, y=weather_risk_vals,
            color=weather_risk_vals,
            color_continuous_scale="Reds",
            labels={"x": "Weather Type", "y": "Mean Risk Severity"},
            title="Accident Risk Severity by Weather Condition",
            template="plotly_dark",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=380,
            font=dict(size=13),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g2:
        risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]
        risk_counts = [48, 34, 18]
        fig_pie = px.pie(
            values=risk_counts, names=risk_labels,
            hole=0.5,
            color=risk_labels,
            color_discrete_map={"Low Risk": "#10B981", "Moderate Risk": "#F59E0B", "High Risk": "#EF4444"},
            title="Predicted Risk Class Distribution Ratio",
            template="plotly_dark",
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.6)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=380,
            font=dict(size=13),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.caption("<span style='font-size:1.0rem;'>🎓 <b>AI Traffic Risk Prediction System</b> | Machine Learning Powered Traffic & Safety Analytics</span>", unsafe_allow_html=True)
