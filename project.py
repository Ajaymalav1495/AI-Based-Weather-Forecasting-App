import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Set Streamlit page config ---
st.set_page_config(page_title="AI Base Weather forecasting App", layout="wide")
st.title("🌦️ AI Based Weather Forecasting & Prediction Platform")

# --- Background (Image or Video) ---
def set_background(image_url=None, video_url=None):
    if image_url:
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)
    elif video_url:
        st.markdown(f"""
            <style>
            .stApp {{
                position: relative;
                overflow: hidden;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: -1;
                background: black;
            }}
            video.background-video {{
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
                z-index: -2;
                object-fit: cover;
            }}
            </style>
            <video class="background-video" autoplay muted loop>
                <source src="{video_url}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)

# Choose one:
set_background(
    image_url= "https://th.bing.com/th/id/R.69515623e0602610c3726c12af746b7c?rik=B%2fDCavopp9n7tQ&riu=http%3a%2f%2fwallpapercave.com%2fwp%2f80cDgU8.jpg&ehk=1Xs3Vr0Fg%2fSxGvO6lI19MxsgNmkBMkgstBMpZEOPw70%3d&risl=&pid=ImgRaw&r=0"
)

# --- Sidebar ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select a Section:",
    ["Current Weather", "7-Time Forecast In a Day ", "Train & Predict Model", "State-Based Forecast Next 7 Day from CSV"]
)

# Sidebar API key
api_key = st.sidebar.text_input(
    "Enter your OpenWeatherMap API Key:",
    value="e572ee77cfae76f6c2105f4305468d42",  # Mask your real key in production
    type="password"
)

# --------------------- Section 1: Current Weather ---------------------
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

if menu == "Current Weather":
    st.header("Current Weather")
    city = st.text_input("Enter city name:")
    if city and api_key:
        weather = fetch_current_weather(city, api_key)
        if weather:
            st.subheader(f"Weather in {city.title()}")
            for key, val in weather.items():
                st.write(f"{key}: {val}")
        else:
            st.error("Failed to fetch weather data. Please check the city name or API key.")

# --------------------- Section 2: 7-Day Forecast ---------------------
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
        "Wind Speed (m/s)": [entry["wind"]["speed"] for entry in forecast],
        "Pressure (hPa)": [entry["main"]["pressure"] for entry in forecast],
        "Precipitation (mm)": [entry.get("rain", {}).get("3h", 0) for entry in forecast],
    })

if menu == "7-Time Forecast In a Day":
    st.header("7-Time Forecast In a Day")
    city = st.text_input("Enter city name for forecast:")
    if city and api_key:
        forecast_df = fetch_weather_forecast(city, api_key)
        if forecast_df is not None:
            st.dataframe(forecast_df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast_df["Datetime"], forecast_df["Temperature (°C)"], marker='o')
            ax.set_title(f"7-Day Temperature Forecast for {city.title()}")
            ax.set_xlabel("Datetime")
            ax.set_ylabel("Temperature (°C)")
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("Error fetching forecast data.")

# --------------------- Section 3: Train and Predict Model ---------------------
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
    return model, X_test, y_test, predictions, mse

if menu == "Train & Predict Model":
    st.header("Train & Predict Weather Model")
    st.info("Upload CSV with columns: 'temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm'")
    uploaded = st.file_uploader("Upload Weather Dataset CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            model, X_test, y_test, predictions, mse = train_weather_model(df)
            st.success(f"Model trained. Mean Squared Error: {mse:.2f}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test.values, label="Actual", marker='o')
            ax.plot(predictions, label="Predicted", linestyle='--', marker='x')
            ax.set_title("Temperature: Actual vs Predicted")
            ax.set_ylabel("Temperature (°C)")
            ax.set_xlabel("Sample Index")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# --------------------- Section 4: Forecast State-Based CSV ---------------------
if menu == "State-Based Forecast Next 7 Day from CSV":
    st.header("State-Based Forecast Next 7 Day from CSV")
    csv_file = st.file_uploader("Upload IndianWeatherRepository.csv", type=["csv"])
    if csv_file:
        data = pd.read_csv(csv_file)
        state_input = st.text_input("Enter the state name:")
        if state_input:
            state = state_input.title()
            state_data = data[data["region"] == state]
            if state_data.empty:
                st.error(f"No data found for State: {state}")
            else:
                st.success(f"Current Temperature in {state}: {state_data['temperature_celsius'].iloc[-1]:.2f}°C")
                features = ["temperature_celsius"]
                target = "temperature_celsius"
                X = state_data[features].fillna(0)
                y = state_data[target].fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                future_data = X_test.sample(7, replace=True)
                predictions = model.predict(future_data)
                st.subheader("Next 7-Day Temperature Forecast")
                for i, temp in enumerate(predictions):
                    st.write(f"Day {i+1}: {temp:.2f}°C")

