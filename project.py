import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from streamlit_option_menu import option_menu

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Weather & Farming Risk App", layout="wide")
st.title("🌍 AI Weather Forecasting & 🌾 Farming Risk Prediction")

# --- Styling ---
st.markdown("""
    <style>
   @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Montserrat', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #ADFF2F;
        font-weight: 700;
    }
    .header {
        font-size: 36px;
        color: #ADFF2F;
        text-align: center;
        margin-top: 30px;
    }
    .subheader {
        font-size: 24px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 20px;
    }
    .btn {
        background-color: #ADFF2F !important;
        color: #000 !important;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: bold;
        text-transform: uppercase;
        margin-top: 20px;
    }
    .card {
        background: #1e1e1e;
        padding: 20px;
        margin: 20px 0;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 255, 0, 0.15);
    }
    .stTextInput>div>div>input {
        background-color: #2c2c2c;
        color: white;
    }
    .stFileUploader, .stTextInput, .stSelectbox, .stNumberInput {
        background-color: #2c2c2c;
        border-radius: 10px;
    }
    .css-1d391kg {
        color: #ADFF2F;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    menu = option_menu(
        menu_title="Navigation",
        options=["Current Weather", "7-Time Forecast", "Train & Predict Model", "State-Based Forecast", "Agri Risk Predictor"],
        icons=["cloud-sun", "calendar3", "cpu", "bar-chart", "activity"],
        menu_icon="cast",
        default_index=0
    )

# Sidebar API Key Input
api_key = "e572ee77cfae76f6c2105f4305468d42"

# --- Function: Current Weather ---
def fetch_current_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        "Temperature (°C)": data["main"]["temp"],
        "Humidity (%)": data["main"]["humidity"],
        "Pressure (hPa)": data["main"]["pressure"],
        "Wind Speed (m/s)": data["wind"]["speed"],
        "Weather Type": data["weather"][0]["description"]
    }

# --- Function: Forecast ---
def fetch_weather_forecast(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    forecast = response.json()["list"][:7]
    return pd.DataFrame({
        "Datetime": [entry["dt_txt"] for entry in forecast],
        "Temperature (°C)": [entry["main"]["temp"] for entry in forecast],
        "Humidity (%)": [entry["main"]["humidity"] for entry in forecast],
        "Wind Speed (m/s)": [entry["wind"]["speed"] for entry in forecast]
    })

# --- Function: Train Model ---
def train_weather_model(data):
    features = ["temperature_celsius", "humidity", "wind_kph", "pressure_mb", "precip_mm"]
    target = "temperature_celsius"
    data.dropna(inplace=True)
    X = data[features]
    y = data[target].shift(-7)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-7], y[:-7], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, predictions, mse

# --- Function: Agri Risk Predictor ---
def predict_agri_risk(temperature, humidity, soil_moisture, pest_level):
    pest_map = {"None": 0, "Mild": 1, "Severe": 2}
    X_sample = pd.DataFrame([[temperature, humidity, soil_moisture, pest_map[pest_level]]], 
                            columns=["Temperature", "Humidity", "Soil Moisture", "Pest Level"])
    
    np.random.seed(42)
    X_train = pd.DataFrame({
        "Temperature": np.random.randint(10, 50, 100),
        "Humidity": np.random.randint(10, 100, 100),
        "Soil Moisture": np.random.randint(20, 80, 100),
        "Pest Level": np.random.randint(0, 3, 100)
    })
    y_train = np.random.choice(["Low Risk", "Moderate Risk", "High Risk"], size=100)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_sample)[0]
    return prediction

# ----------------------- Sections -----------------------

# 1. Current Weather
if menu == "Current Weather":
    st.header("🌍 Current Weather")
    city = st.text_input("Enter City:")
    if city and api_key:
        weather = fetch_current_weather(city, api_key)
        if weather:
            st.subheader(f"Weather in {city.title()}")
            st.write(weather)
        else:
            st.error("Failed to fetch weather data.")

# 2. Forecast
elif menu == "7-Time Forecast":
    st.header("📅 7-Time Forecast")
    city = st.text_input("Enter City:")
    if city and api_key:
        forecast_df = fetch_weather_forecast(city, api_key)
        if forecast_df is not None:
            st.dataframe(forecast_df.style.set_properties(**{"background": "#333", "color": "white"}))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast_df["Datetime"], forecast_df["Temperature (°C)"], marker='o', color="lime")
            ax.set_title(f"Temperature Forecast for {city.title()}")
            ax.set_xlabel("Datetime")
            ax.set_ylabel("Temp (°C)")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("Error fetching forecast.")

# 3. Train & Predict Model
elif menu == "Train & Predict Model":
    st.header("🔬 Train Model with CSV")
    uploaded = st.file_uploader("Upload Weather CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        model, predictions, mse = train_weather_model(df)
        st.success(f"Model trained. MSE: {mse:.2f}")

# 4. State-Based Forecast
elif menu == "State-Based Forecast":
    st.header("📍 Forecast by State")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file:
        data = pd.read_csv(csv_file)
        state = st.text_input("Enter State:")
        if state:
            match = data[data["region"].str.title() == state.title()]
            if not match.empty:
                st.success(f"🌡 Temp in {state}: {match['temperature_celsius'].iloc[-1]:.2f}°C")
            else:
                st.error("No data found for that state.")

# 5. Agri Risk Predictor
elif menu == "Agri Risk Predictor":
    st.header("🌾 Agricultural Risk Predictor")
    city = st.text_input("Enter City for Weather:")
    if city and api_key:
        weather_data = fetch_current_weather(city, api_key)
        if weather_data:
            st.success("Weather Data Retrieved")
            st.write(weather_data)
            soil = np.random.randint(20, 80)
            pest = np.random.choice(["None", "Mild", "Severe"])
            st.write(f"Soil Moisture: {soil}%")
            st.write(f"Pest Level: {pest}")
            risk = predict_agri_risk(weather_data["Temperature (°C)"], weather_data["Humidity (%)"], soil, pest)
            st.success(f"🌿 Predicted Agri Risk: **{risk}**")
        else:
            st.error("Failed to fetch weather data.")
