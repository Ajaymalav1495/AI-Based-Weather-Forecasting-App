import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Weather & Farming Risk App", layout="wide")
st.title("🌍 AI Weather Forecasting & 🌾 Farming Risk Prediction")

# --- Sidebar ---
with st.sidebar:
    dark_mode = st.checkbox("🌗 Dark Mode", value=True)
    menu = option_menu(
        menu_title="Navigation",
        options=["Current Weather", "7-Time Forecast", "Train & Predict Model", "State-Based Forecast", "Agri Risk Predictor"],
        icons=["cloud-sun", "calendar3", "cpu", "bar-chart", "activity"],
        menu_icon="cast",
        default_index=0
    )

# --- Theme Setup ---
if dark_mode:
    background = "#121212"
    text_color = "#e0e0e0"
    header_color = "#ADFF2F"
    card_bg = "#1e1e1e"
    input_bg = "#2c2c2c"
    box_shadow = "rgba(0, 255, 0, 0.15)"
    background_image = "background-image: url('https://ibb.co/sdB9X9RL');"
else:
    background = "#f9f9f9"
    text_color = "#000"
    header_color = "#000"
    card_bg = "#ffffff"
    input_bg = "#ffffff"
    box_shadow = "rgba(0, 0, 0, 0.1)"
    background_image = ""

# --- Dynamic Styling ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    .stApp {{
        background-color: {background};
        {background_image}
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: {text_color};
        font-family: 'Montserrat', sans-serif;
    }}
    h1, h2, h3, h4 {{
        color: {header_color};
        font-weight: 700;
    }}
    .btn {{
        background-color: #ADFF2F !important;
        color: #000 !important;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: bold;
        text-transform: uppercase;
        margin-top: 20px;
    }}
    .card {{
        background: {card_bg};
        padding: 20px;
        margin: 20px 0;
        border-radius: 15px;
        box-shadow: 0px 4px 10px {box_shadow};
    }}
    .stTextInput>div>div>input {{
        background-color: {input_bg};
        color: {text_color};
    }}
    .stFileUploader, .stTextInput, .stSelectbox, .stNumberInput {{
        background-color: {input_bg};
        border-radius: 10px;
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# --- API Key ---
api_key = "e572ee77cfae76f6c2105f4305468d42"

# --- Functions ---
def fetch_current_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return pd.DataFrame({
        "Temperature (°C)": [data["main"]["temp"]],
        "Humidity (%)": [data["main"]["humidity"]],
        "Pressure (hPa)": [data["main"]["pressure"]],
        "Wind Speed (m/s)": [data["wind"]["speed"]],
        "Weather Type": [data["weather"][0]["description"]]
    })

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
    return model.predict(X_sample)[0]

# --- App Sections ---
if menu == "Current Weather":
    st.header("🌍 Current Weather")
    city = st.text_input("Enter City:")
    if city:
        weather = fetch_current_weather(city, api_key)
        if weather is not None:
            st.subheader(f"Weather in {city.title()}")
            st.write(weather)
        else:
            st.error("Failed to fetch weather data.")

elif menu == "7-Time Forecast":
    st.header("📅 7-Time Forecast")
    city = st.text_input("Enter City:")
    if city:
        forecast_df = fetch_weather_forecast(city, api_key)
        if forecast_df is not None:
            st.dataframe(forecast_df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast_df["Datetime"], forecast_df["Temperature (°C)"], marker='o', color="lime")
            ax.set_title(f"Temperature Forecast for {city.title()}")
            ax.set_xlabel("Datetime")
            ax.set_ylabel("Temp (°C)")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("Error fetching forecast.")

elif menu == "Train & Predict Model":
    st.header("🔬 Train Model with CSV")
    uploaded = st.file_uploader("Upload Weather CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        model, predictions, mse = train_weather_model(df)
        st.success(f"Model trained. MSE: {mse:.2f}")

elif menu == "State-Based Forecast":
    st.header("📍 State-Based 7-Day Forecast")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    state = st.text_input("Enter State:")
    if state:
        forecast_data = None
        if csv_file:
            data = pd.read_csv(csv_file)
            forecast_data = data[data["region"].str.title() == state.title()]
            if forecast_data.empty:
                st.error("No data found for the state in the uploaded file.")
                forecast_data = None
        if forecast_data is None:
            city = st.text_input("Enter a city within the state for forecast:")
            if city:
                forecast_df = fetch_weather_forecast(city, api_key)
                if forecast_df is not None:
                    forecast_data = forecast_df
                else:
                    st.error("Error fetching forecast.")
        if forecast_data is not None:
            st.subheader(f"7-Day Weather Forecast for {state.title()}")
            st.dataframe(forecast_data)
            if "Datetime" in forecast_data and "Temperature (°C)" in forecast_data:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(forecast_data["Datetime"], forecast_data["Temperature (°C)"], marker='o', color="lime")
                ax.set_title(f"7-Day Temperature Trend for {state.title()}")
                ax.set_xlabel("Datetime")
                ax.set_ylabel("Temperature (°C)")
                plt.xticks(rotation=45)
                st.pyplot(fig)

elif menu == "Agri Risk Predictor":
    st.header("🌾 Agricultural Risk Predictor")
    city = st.text_input("Enter City for Weather:")
    if city:
        weather_data = fetch_current_weather(city, api_key)
        if weather_data is not None:
            st.success("Weather Data Retrieved")
            st.write(weather_data)
            soil = np.random.randint(20, 80)
            pest = np.random.choice(["None", "Mild", "Severe"])
            st.write(f"Soil Moisture: {soil}%")
            st.write(f"Pest Level: {pest}")
            risk = predict_agri_risk(weather_data["Temperature (°C)"][0], weather_data["Humidity (%)"][0], soil, pest)
            st.success(f"🌿 Predicted Agri Risk: **{risk}**")
        else:
            st.error("Failed to fetch weather data.")
