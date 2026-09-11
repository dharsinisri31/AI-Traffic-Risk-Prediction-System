import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from typing import Dict, Any, Tuple

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="AI Traffic Risk & Congestion Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BACKEND_URL = API_URL

# Custom CSS for modern, professional dark-navy & glass card aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    /* Background & Main Canvas */
    .stApp {
        background: radial-gradient(circle at top right, #111e38 0%, #080c16 60%, #05070d 100%);
        color: #f1f5f9;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0c1222;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Custom Sidebar Brand Card */
    .brand-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .brand-icon {
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.5);
    }
    .brand-title {
        font-size: 16px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Sleek Status Pill */
    .status-pill-online {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        margin-top: 10px;
        width: 100%;
        justify-content: center;
    }
    .status-pill-offline {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        margin-top: 10px;
        width: 100%;
        justify-content: center;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    .pulse-dot-red {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ef4444;
        box-shadow: 0 0 8px #ef4444;
    }
    
    /* Sidebar Radio Customization */
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #cbd5e1;
        font-weight: 500;
        font-size: 13.5px;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(59, 130, 246, 0.12);
        border-color: rgba(59, 130, 246, 0.3);
        color: #ffffff;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.25) 0%, rgba(14, 165, 233, 0.15) 100%) !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.25);
    }

    /* Top Banner Header */
    .header-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 58, 138, 0.7) 60%, rgba(2, 132, 199, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 24px 30px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.4px;
    }
    .header-banner p {
        color: #94a3b8;
        font-size: 13.5px;
        margin: 6px 0 0 0;
    }
    
    /* Modern Glass Card */
    .glass-panel {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(180deg, rgba(26, 38, 66, 0.6) 0%, rgba(13, 20, 36, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        margin: 8px 0 4px 0;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        color: #64748b;
        font-size: 12px;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(26, 38, 66, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.4);
    }
    
    /* Risk Badge Styles */
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.4);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        font-size: 12px;
        color: #94a3b8;
        margin-top: 16px;
        line-height: 1.5;
    }

    /* Connection alert */
    .backend-error-box {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 16px 20px;
        border-radius: 14px;
        color: #fca5a5;
        margin-bottom: 20px;
        font-size: 13.5px;
    }

    /* Primary Submit Button Glow */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type": "Traffic Congestion",
            "prediction": "High",
            "confidence": 0.88,
            "weather": "Rain",
            "hour": 18,
            "risk_score": 88.0
        },
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type": "Accident Risk",
            "prediction": "Moderate",
            "confidence": 0.58,
            "weather": "Clouds",
            "hour": 14,
            "risk_score": 58.2
        },
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type": "Traffic Congestion",
            "prediction": "Low",
            "confidence": 0.76,
            "weather": "Clear",
            "hour": 10,
            "risk_score": 24.0
        }
    ]

# ==========================================
# API HELPER FUNCTIONS
# ==========================================
def check_backend_status() -> Tuple[bool, str]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            return True, "Online"
        return False, "Offline"
    except requests.exceptions.RequestException:
        return False, "Offline"

def call_predict_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{BACKEND_URL}/{endpoint}", json=payload, timeout=6)
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        else:
            detail = r.json().get("detail", f"Server error ({r.status_code})")
            return {"success": False, "error": detail}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Backend API is not running. Start FastAPI using: uvicorn backend.main:app --reload"
        }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out while contacting prediction service."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# ==========================================
# SIDEBAR NAVIGATION (CLEAN & MODERN)
# ==========================================
with st.sidebar:
    # Sleek Branding Banner
    st.markdown("""
    <div class="brand-card">
        <div class="brand-icon">🚦</div>
        <div>
            <div class="brand-title">TrafficAI Core</div>
            <div class="brand-subtitle">Risk & Mobility System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px 4px;'>Navigation</p>", unsafe_allow_html=True)
    
    nav_options = [
        "📊 Dashboard",
        "🚗 Congestion Prediction",
        "⚠️ Accident Risk Estimation",
        "📈 Analytics & Trends",
        "ℹ️ System Architecture"
    ]
    
    selected_nav = st.radio(
        "Navigation Menu",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )
    
    # Map back to logical page names
    page_map = {
        "📊 Dashboard": "Dashboard",
        "🚗 Congestion Prediction": "Traffic Congestion Prediction",
        "⚠️ Accident Risk Estimation": "Accident Risk Prediction",
        "📈 Analytics & Trends": "Analytics",
        "ℹ️ System Architecture": "About"
    }
    page = page_map[selected_nav]
    
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    # Minimalist System Health Badge (No port numbers or clutter)
    is_online, status_text = check_backend_status()
    if is_online:
        st.markdown("""
        <div class="status-pill-online">
            <span class="pulse-dot"></span> AI Engine Online
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pill-offline">
            <span class="pulse-dot-red"></span> AI Engine Offline
        </div>
        """, unsafe_allow_html=True)
        st.caption("<div style='text-align:center; font-size:11px; color:#94a3b8; margin-top:4px;'>Run: <code>uvicorn backend.main:app --reload</code></div>", unsafe_allow_html=True)

# Global Header
st.markdown("""
<div class="header-banner">
    <h1>AI-Based Traffic Congestion & Accident Risk Prediction System</h1>
    <p>Decision-Support System powered by Machine Learning & Real-Time Environmental Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

if not is_online:
    st.markdown("""
    <div class="backend-error-box">
        <strong>⚠️ Backend API Notice:</strong> Backend API is currently unreachable. Start FastAPI using: <code>uvicorn backend.main:app --reload</code>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# CATEGORICAL LISTS (From Trained Schema)
# ==========================================
HOLIDAYS = [
    "None", "Columbus Day", "Independence Day", "Labor Day",
    "Martin Luther King Jr Day", "Memorial Day", "New Years Day",
    "State Fair", "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"
]

WEATHER_MAINS = [
    "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
]

WEATHER_DESCRIPTIONS = [
    "Sky is Clear", "broken clouds", "drizzle", "few clouds", "fog", "freezing rain", "haze",
    "heavy intensity drizzle", "heavy intensity rain", "heavy snow", "light intensity drizzle",
    "light intensity shower rain", "light rain", "light rain and snow", "light shower snow",
    "light snow", "mist", "moderate rain", "overcast clouds", "proximity shower rain",
    "proximity thunderstorm", "proximity thunderstorm with drizzle", "proximity thunderstorm with rain",
    "scattered clouds", "shower drizzle", "shower snow", "sky is clear", "sleet", "smoke",
    "snow", "thunderstorm", "thunderstorm with drizzle", "thunderstorm with heavy rain",
    "thunderstorm with light drizzle", "thunderstorm with light rain", "thunderstorm with rain",
    "very heavy rain"
]

# ==========================================
# 1. DASHBOARD PAGE
# ==========================================
if page == "Dashboard":
    st.subheader("System Overview & Live Performance Metrics")
    
    total_preds = len(st.session_state.history)
    latest_cong = next((h for h in reversed(st.session_state.history) if h["type"] == "Traffic Congestion"), None)
    latest_acc = next((h for h in reversed(st.session_state.history) if h["type"] == "Accident Risk"), None)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Predictions</div>
            <div class="metric-value">{total_preds}</div>
            <div class="metric-sub">Processed in active session</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        cong_val = latest_cong['prediction'] if latest_cong else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Congestion</div>
            <div class="metric-value">{cong_val}</div>
            <div class="metric-sub">Confidence: {f"{latest_cong['confidence']*100:.1f}%" if latest_cong else '—'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        acc_val = latest_acc['prediction'] if latest_acc else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Accident Risk</div>
            <div class="metric-value">{acc_val}</div>
            <div class="metric-sub">AI Risk Classification</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        acc_pct = f"{latest_acc['risk_score']:.1f}%" if latest_acc else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Estimated Risk %</div>
            <div class="metric-value">{acc_pct}</div>
            <div class="metric-sub">Probability index estimation</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### 📈 Prediction Confidence & Risk Trend")
        df_hist = pd.DataFrame(st.session_state.history)
        if not df_hist.empty:
            fig = px.line(
                df_hist,
                x=range(1, len(df_hist) + 1),
                y="risk_score",
                color="type",
                markers=True,
                labels={"x": "Prediction #", "risk_score": "Confidence / Risk Index (%)"},
                template="plotly_dark",
                color_discrete_map={"Traffic Congestion": "#38bdf8", "Accident Risk": "#f43f5e"}
            )
            fig.update_layout(
                paper_bgcolor="rgba(17, 24, 39, 0.6)",
                plot_bgcolor="rgba(17, 24, 39, 0.6)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data logged yet.")
            
    with c2:
        st.markdown("##### 🎯 Congestion & Hazard Class Breakdown")
        if not df_hist.empty:
            fig_pie = px.pie(
                df_hist,
                names="prediction",
                hole=0.45,
                color="prediction",
                color_discrete_map={"Low": "#10b981", "Moderate": "#f59e0b", "Medium": "#f59e0b", "High": "#ef4444"},
                template="plotly_dark"
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(17, 24, 39, 0.6)",
                plot_bgcolor="rgba(17, 24, 39, 0.6)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No distribution data available.")

    st.markdown("##### 📋 Recent Activity Log")
    if not df_hist.empty:
        st.dataframe(
            df_hist[["timestamp", "type", "prediction", "confidence", "weather", "hour", "risk_score"]].tail(5),
            use_container_width=True
        )

# ==========================================
# 2. TRAFFIC CONGESTION PREDICTION PAGE
# ==========================================
elif page == "Traffic Congestion Prediction":
    st.subheader("🚗 Traffic Congestion Level Prediction")
    st.caption("Provide environmental and temporal features to analyze road congestion probability using the trained Random Forest Classifier.")

    with st.form("congestion_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Temporal Features**")
            p_date = st.date_input("Date", datetime.now())
            p_time = st.time_input("Time", dtime(17, 0))
            is_peak = st.checkbox("Peak Commute Hour", value=True)
            holiday_selected = st.selectbox("Holiday", HOLIDAYS, index=0)
            
        with col2:
            st.markdown("**🌤️ Environmental & Weather**")
            temp_c = st.slider("Temperature (°C)", min_value=-30.0, max_value=50.0, value=22.0, step=0.5)
            clouds = st.slider("Cloud Coverage (%)", min_value=0, max_value=100, value=40)
            rain = st.number_input("Rainfall in past 1h (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            snow = st.number_input("Snowfall in past 1h (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            
        with col3:
            st.markdown("**⛈️ Weather Description & Risk Indices**")
            weather_main = st.selectbox("Weather Main Category", WEATHER_MAINS, index=0)
            weather_desc = st.selectbox("Weather Condition Detail", WEATHER_DESCRIPTIONS, index=3)
            traffic_risk_idx = st.slider("Traffic Congestion Risk Index (0-5)", 0.0, 5.0, 2.0, 0.5)
            weather_risk_idx = st.slider("Weather Severity Risk Index (0-5)", 0.0, 5.0, 1.0, 0.5)

        submit_cong = st.form_submit_button("Predict Congestion", use_container_width=True, type="primary")

    if submit_cong:
        # Build payload matching 73 feature columns exactly
        temp_k = temp_c + 273.15
        hour_val = p_time.hour
        day_val = p_date.day
        month_val = p_date.month
        year_val = p_date.year
        dow_val = p_date.weekday()
        is_weekend_val = 1 if dow_val in [5, 6] else 0

        payload = {
            "temp": temp_k,
            "rain_1h": float(rain),
            "snow_1h": float(snow),
            "clouds_all": int(clouds),
            "hour": int(hour_val),
            "day": int(day_val),
            "month": int(month_val),
            "year": int(year_val),
            "day_of_week": int(dow_val),
            "is_weekend": int(is_weekend_val),
            "is_peak_hour": 1 if is_peak else 0,
            "traffic_risk": float(traffic_risk_idx),
            "rain_risk": 1.0 if rain > 0 else 0.0,
            "snow_risk": 1.0 if snow > 0 else 0.0,
            "weather_risk": float(weather_risk_idx),
        }
        
        # Add one-hot encoded flags
        payload[f"holiday_{holiday_selected}"] = 1
        payload[f"weather_main_{weather_main}"] = 1
        payload[f"weather_description_{weather_desc}"] = 1

        with st.spinner("Analyzing traffic congestion patterns..."):
            res = call_predict_api("predict/congestion", payload)

        if res["success"]:
            data = res["data"]
            level = data.get("congestion_level", "Unknown")
            conf = data.get("confidence")
            rec = data.get("recommendation", "")

            # Badge styling
            badge_class = "badge-low" if level == "Low" else ("badge-moderate" if level in ("Medium", "Moderate") else "badge-high")

            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">Prediction Result</span>
                    <span class="{badge_class}">{level.upper()} CONGESTION</span>
                </div>
                <div style="margin: 16px 0;">
                    <div style="font-size: 30px; font-weight: 800; color: #ffffff;">{level} Traffic Congestion</div>
                    <div style="font-size: 14px; color: #cbd5e1; margin-top: 4px;">
                        Model Confidence: <strong>{f"{conf*100:.1f}%" if conf is not None else "N/A"}</strong>
                    </div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.04); padding: 14px 18px; border-radius: 12px; border-left: 4px solid #38bdf8;">
                    <div style="font-size: 11px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Actionable Recommendation</div>
                    <div style="font-size: 14px; color: #f8fafc; margin-top: 4px; line-height: 1.5;">{rec}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Store in session state
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": "Traffic Congestion",
                "prediction": level,
                "confidence": conf if conf is not None else 0.0,
                "weather": weather_main,
                "hour": hour_val,
                "risk_score": (conf * 100.0) if conf is not None else 50.0
            })
        else:
            st.error(res["error"])

# ==========================================
# 3. ACCIDENT RISK PREDICTION PAGE
# ==========================================
elif page == "Accident Risk Prediction":
    st.subheader("⚠️ Accident Risk Estimation")
    st.caption("AI-powered decision-support risk model estimating situational hazard probabilities from environmental and road factors.")

    with st.form("accident_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Date & Commute Window**")
            p_date = st.date_input("Date", datetime.now(), key="acc_date")
            p_time = st.time_input("Time", dtime(18, 30), key="acc_time")
            is_peak = st.checkbox("Peak Traffic Hours", value=True, key="acc_peak")
            holiday_selected = st.selectbox("Holiday", HOLIDAYS, index=0, key="acc_holiday")
            
        with col2:
            st.markdown("**🌧️ Hazardous Weather Conditions**")
            temp_c = st.slider("Temperature (°C)", -30.0, 50.0, 4.0, 0.5, key="acc_temp")
            clouds = st.slider("Cloud Coverage (%)", 0, 100, 85, key="acc_clouds")
            rain = st.number_input("Rainfall (mm/h)", min_value=0.0, max_value=100.0, value=8.5, step=0.1, key="acc_rain")
            snow = st.number_input("Snowfall (mm/h)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, key="acc_snow")
            
        with col3:
            st.markdown("**🚨 Risk Sensitivity Indices**")
            weather_main = st.selectbox("Weather Main Category", WEATHER_MAINS, index=5, key="acc_wmain")
            weather_desc = st.selectbox("Detailed Weather Condition", WEATHER_DESCRIPTIONS, index=8, key="acc_wdesc")
            traffic_risk_idx = st.slider("Traffic Congestion Risk (0-5)", 0.0, 5.0, 3.5, 0.5, key="acc_trisk")
            weather_risk_idx = st.slider("Weather Hazard Index (0-5)", 0.0, 5.0, 3.0, 0.5, key="acc_wrisk")

        submit_acc = st.form_submit_button("Predict Accident Risk", use_container_width=True, type="primary")

    if submit_acc:
        temp_k = temp_c + 273.15
        hour_val = p_time.hour
        day_val = p_date.day
        month_val = p_date.month
        year_val = p_date.year
        dow_val = p_date.weekday()
        is_weekend_val = 1 if dow_val in [5, 6] else 0

        payload = {
            "temp": temp_k,
            "rain_1h": float(rain),
            "snow_1h": float(snow),
            "clouds_all": int(clouds),
            "hour": int(hour_val),
            "day": int(day_val),
            "month": int(month_val),
            "year": int(year_val),
            "day_of_week": int(dow_val),
            "is_weekend": int(is_weekend_val),
            "is_peak_hour": 1 if is_peak else 0,
            "traffic_risk": float(traffic_risk_idx),
            "rain_risk": 2.0 if rain > 5.0 else (1.0 if rain > 0 else 0.0),
            "snow_risk": 2.0 if snow > 2.0 else (1.0 if snow > 0 else 0.0),
            "weather_risk": float(weather_risk_idx),
        }
        payload[f"holiday_{holiday_selected}"] = 1
        payload[f"weather_main_{weather_main}"] = 1
        payload[f"weather_description_{weather_desc}"] = 1

        with st.spinner("Computing accident risk probability metrics..."):
            res = call_predict_api("predict/accident", payload)

        if res["success"]:
            data = res["data"]
            risk_lvl = data.get("risk_level", "Moderate")
            risk_pct = data.get("risk_percentage", 50.0)
            rec = data.get("recommendation", "")

            # Determine risk tier based on percentage or level
            if risk_pct <= 32.0:
                tier_label = "Low"
                badge_class = "badge-low"
                gauge_color = "#10b981"
            elif risk_pct <= 65.0:
                tier_label = "Moderate"
                badge_class = "badge-moderate"
                gauge_color = "#f59e0b"
            else:
                tier_label = "High"
                badge_class = "badge-high"
                gauge_color = "#ef4444"

            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">AI Accident Risk Estimation</span>
                    <span class="{badge_class}">{risk_lvl.upper()} RISK ({risk_pct:.1f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk Gauge Indicator
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Level: {risk_lvl}", 'font': {'size': 18, 'color': '#ffffff'}},
                number={'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#374151",
                    'steps': [
                        {'range': [0, 32], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [32, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': risk_pct
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(17, 24, 39, 0.6)",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.04); padding: 16px 20px; border-radius: 12px; border-left: 4px solid {gauge_color}; margin-top: 10px;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Safety Recommendation</div>
                <div style="font-size: 14.5px; color: #f8fafc; margin-top: 4px; line-height: 1.5;">{rec}</div>
            </div>
            
            <div class="disclaimer-box">
                ℹ️ <strong>Decision-Support Notice:</strong> This prediction is an AI-based accident risk estimation intended to assist traffic management and situational awareness. It does not guarantee whether an actual collision will occur.
            </div>
            """, unsafe_allow_html=True)

            # Store in session state
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": "Accident Risk",
                "prediction": risk_lvl,
                "confidence": risk_pct / 100.0,
                "weather": weather_main,
                "hour": hour_val,
                "risk_score": float(risk_pct)
            })
        else:
            st.error(res["error"])

# ==========================================
# 4. ANALYTICS PAGE
# ==========================================
elif page == "Analytics":
    st.subheader("📊 Model Performance & Prediction Analytics")
    st.caption("Aggregated analytics and historical trend analysis across all user query sessions.")

    df_hist = pd.DataFrame(st.session_state.history)
    
    if df_hist.empty:
        st.info("No prediction records found in the current session. Run predictions from the sidebar to populate analytics.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🚦 Congestion Prediction Breakdown")
            df_cong = df_hist[df_hist["type"] == "Traffic Congestion"]
            if not df_cong.empty:
                fig_c = px.histogram(
                    df_cong,
                    x="prediction",
                    color="prediction",
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                    template="plotly_dark"
                )
                fig_c.update_layout(paper_bgcolor="rgba(17, 24, 39, 0.6)", plot_bgcolor="rgba(17, 24, 39, 0.6)", height=300)
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("No congestion predictions yet.")

        with col2:
            st.markdown("##### ⚠️ Accident Risk Distribution")
            df_acc = df_hist[df_hist["type"] == "Accident Risk"]
            if not df_acc.empty:
                fig_a = px.pie(
                    df_acc,
                    names="prediction",
                    color="prediction",
                    color_discrete_map={"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"},
                    template="plotly_dark",
                    hole=0.45
                )
                fig_a.update_layout(paper_bgcolor="rgba(17, 24, 39, 0.6)", plot_bgcolor="rgba(17, 24, 39, 0.6)", height=300)
                st.plotly_chart(fig_a, use_container_width=True)
            else:
                st.info("No accident risk estimations yet.")

        st.markdown("##### 📈 Risk Percentage Timeline & Severity")
        fig_timeline = px.scatter(
            df_hist,
            x="timestamp",
            y="risk_score",
            color="prediction",
            size=[14]*len(df_hist),
            hover_data=["type", "weather", "hour"],
            template="plotly_dark",
            color_discrete_map={"Low": "#10b981", "Moderate": "#f59e0b", "Medium": "#f59e0b", "High": "#ef4444"}
        )
        fig_timeline.update_layout(paper_bgcolor="rgba(17, 24, 39, 0.6)", plot_bgcolor="rgba(17, 24, 39, 0.6)", height=320)
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("##### 🗄️ Full Prediction History Ledger")
        st.dataframe(df_hist, use_container_width=True)

# ==========================================
# 5. ABOUT PAGE
# ==========================================
elif page == "About":
    st.subheader("ℹ️ Project Architecture & Documentation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Problem Statement & Objective
        Urban traffic congestion and road accident hazards contribute heavily to economic losses, increased carbon emissions, and safety risks worldwide. 
        
        The primary objective of this project is to build an end-to-end, intelligent decision-support system capable of:
        1. Accurately classifying road **traffic congestion severity** (Low, Medium, High).
        2. Estimating localized **accident risk probability** (Low, Moderate, High) across varying environmental and temporal conditions.
        """)
        
        st.markdown("""
        ### 📊 Dataset & Feature Engineering
        - **Source**: Comprehensive traffic volume, weather station observations, and calendar event data.
        - **Engineered Features**: Peak commute hour flags, weekend indicators, one-hot encoded holiday factors, and weather risk severity indices.
        - **Total Model Features**: **73 aligned feature columns** preserved via `joblib` PKL schemas.
        """)

    with col2:
        st.markdown("""
        ### 🧠 Machine Learning Models
        - **Traffic Congestion Model**: `RandomForestClassifier` trained for non-linear multi-class categorization with confidence estimation.
        - **Accident Risk Estimation Model**: `GradientBoostingClassifier` fine-tuned for risk sensitivity classification across 3 distinct hazard bands (0–32% Low, 33–65% Moderate, 66–100% High).
        """)

        st.markdown("""
        ### 🛠️ Architecture & Deployment
        - **Backend Framework**: **FastAPI** with asynchronous request routing, Pydantic data validation, and singleton model management.
        - **Frontend Interface**: **Streamlit** dashboard styled with custom CSS and Plotly data visualizations.
        - **PKL Deployment**: Production estimators loaded once at startup to guarantee low-latency inference.
        """)

    st.divider()
    
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        ### ⚠️ System Limitations
        - Predictions are probabilistic decision-support estimates and do not guarantee the occurrence or absence of an accident.
        - Relies on accurate weather and temporal input sensor feeds.
        """)
    with c4:
        st.markdown("""
        ### 🚀 Future Enhancements
        - Integration of live GPS / OpenStreetMap traffic sensor telemetry.
        - Dynamic deep learning models (e.g. Spatio-Temporal Graph Neural Networks / LSTM).
        - Automated emergency response routing suggestions.
        """)
