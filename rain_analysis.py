#Universidad Autónoma de Chihuahua
#Machine Learning
#Proyecto Final - Rain in Australia
#Gael Aristides Hinojos Ramírez

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import xgboost

def load():
    log_reg = joblib.load('logistic_regression.sav')
    xgb = joblib.load('xgb_ensemble.sav')
    adab = joblib.load('adaboost_model.sav')
    pca = joblib.load('pca.sav')
    pipeline = joblib.load('pipeline.sav')
    return (pipeline, pca, log_reg, adab, xgb)

def predict(data):
    pipeline, pca, log_reg, adab, xgb = load()
    pipelined_data = pipeline.transform(data)
    pipelined_data = pca.transform(pipelined_data)
    pred_log = log_reg.predict(pipelined_data)
    pred_ad = adab.predict(pipelined_data)
    pred_xgb = xgb.predict(pipelined_data)
    return (pred_log[0], pred_ad[0], pred_xgb[0])
    
st.title("¿Va a llover mañana?")
st.header("Variables")
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle',
        'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport',
        'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong',
        'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne',
        'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
        'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera',
        'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums',
        'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine',
        'Uluru'])
    min_temp = st.number_input(label="Temperatura mínima")
    max_temp = st.number_input(label="Temperatura máxima")
    rainfall = st.number_input(label="Lluvia (mm)")
    evaporation = st.number_input(label="Evaporación (mm)")
    sunshine = st.number_input(label="Horas de luz del sol")
    wind_gust_dir = st.text_input("Dirección de la ráfaga más fuerte")
    wind_gust_speed = st.number_input(label="Velocidad de la ráfaga más fuerte (km/h)")
    wind_dir_9am = st.text_input("Dirección del viento a las 9 a.m.")
    wind_speed_9am = st.number_input(label="Velocidad del viento a las 9 a.m. (km/h)")
    wind_dir_3pm = st.text_input("Dirección del viento a las 3 p.m.")
    wind_speed_3pm = st.number_input(label="Velocidad del viento a las 3 p.m. (km/h)")

with col2:
    humidity_9am = st.number_input(label="Porcentaje de humedad a las 9 a.m.")
    humidity_3pm = st.number_input(label="Porcentaje de humedad a las 3 p.m.")
    pressure_9am = st.number_input(label="Presión atmosférica a las 9 a.m. (hpa)")
    pressure_3pm = st.number_input(label="Presión atmosférica a las 3 p.m. (hpa)")
    cloud_9am = st.number_input(label="Fracción del cielo oscurecido por nubes a las 9 a.m.(oktas)")
    cloud_3pm = st.number_input(label="Fracción del cielo oscurecido por nubes a las 3 p.m. (oktas)")
    temp_9am = st.number_input(label="Temperatura a las 9 a.m.")
    temp_3pm = st.number_input(label="Temperatura a las 3 p.m.")
    day = st.number_input(label="Día")
    month = st.number_input(label="Mes")
    year = st.number_input(label="Año")
    rain_today = st.selectbox("¿LLovió hoy?", ['Yes', 'No'])


if st.button("¿Va a llover mañana?"):
    """data = {'slc': sep_len_cm, 'swc': sep_wid_cm, 'plc': pet_len_cm, 'pwc': pet_wid_cm}
    df = pd.DataFrame([list(data.values())], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])"""
    data = [
        location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir, wind_gust_speed, wind_dir_9am,
        wind_dir_3pm, wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
        cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today, day, month, year
    ]
    df = pd.DataFrame([data], columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir",
                                        "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
                                        "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
                                        "RainToday", "Day", "Month", "Year"])
    pred_log, pred_ad, pred_xgb = predict(df)
    st.text("¿Lloverá mañana según Logistic Regression? {}".format(pred_log))
    st.text("¿Lloverá mañana según Adaboost? {}".format(pred_ad))
    st.text("¿Lloverá mañana según XGBoost? {}".format(pred_xgb))