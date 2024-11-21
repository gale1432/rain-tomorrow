#Universidad Autónoma de Chihuahua
#Machine Learning
#Proyecto Final - Rain in Australia
#Gael Aristides Hinojos Ramírez

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin
import hashlib

#Se crea un Imputador que convierte las columnas de tipo String a valores hash
class StringHasher(BaseEstimator, TransformerMixin):
    def hashString(self, s):
        s = str(s)
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**4
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        new_X = X.copy()
        new_X['Location'] = new_X['Location'].apply(self.hashString)
        new_X['WindDir3pm'] = new_X['WindDir3pm'].apply(self.hashString)
        new_X['WindDir9am'] = new_X['WindDir9am'].apply(self.hashString)
        new_X['WindGustDir'] = new_X['WindGustDir'].apply(self.hashString)
        return new_X
    
#Se crea un Imputador que calcula la media por Localidad
class MeanByLocation(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.dict_mean = {}
        new_X = []
        super().__init__()
    def mean_filler(self, row):
        #print(row['Location'])
        values_fill = self.dict_mean[row['Location']]
        row['MinTemp'] = (values_fill['MinTemp'] if pd.isna(row['MinTemp']) else row['MinTemp'])
        row['MaxTemp']  = (values_fill['MaxTemp'] if pd.isna(row['MaxTemp']) else row['MaxTemp'])
        row['Rainfall']  = (values_fill['Rainfall'] if pd.isna(row['Rainfall']) else row['Rainfall'])
        row['WindGustSpeed'] = (values_fill['WindGustSpeed'] if pd.isna(row['WindGustSpeed']) else row['WindGustSpeed'])
        row['WindSpeed9am'] = (values_fill['WindSpeed9am'] if pd.isna(row['WindSpeed9am']) else row['WindSpeed9am'])
        row['WindSpeed3pm'] = (values_fill['WindSpeed3pm'] if pd.isna(row['WindSpeed3pm']) else row['WindSpeed3pm'])
        row['Humidity9am'] = (values_fill['Humidity9am'] if pd.isna(row['Humidity9am']) else row['Humidity9am'])
        row['Humidity3pm'] = (values_fill['Humidity3pm'] if pd.isna(row['Humidity3pm']) else row['Humidity3pm'])
        row['Pressure9am'] = (values_fill['Pressure9am'] if pd.isna(row['Pressure9am']) else row['Pressure9am'])
        row['Pressure3pm'] = (values_fill['Pressure3pm'] if pd.isna(row['Pressure3pm']) else row['Pressure3pm'])
        row['Temp9am'] = (values_fill['Temp9am'] if pd.isna(row['Temp9am']) else row['Temp9am'])
        row['Temp3pm'] = (values_fill['Temp3pm'] if pd.isna(row['Temp3pm']) else row['Temp3pm'])
        return row
    def fit(self, X, y=None):
        self.new_X = X.copy()
        self.dict_mean =  self.new_X.groupby('Location').mean().to_dict('index')
        print(self.dict_mean)
        return self
    def transform(self, X):
        real_X = X.copy()
        real_X = real_X.apply(self.mean_filler, axis=1)
        return real_X

def load():
    log_reg = joblib.load('logistic_regression.sav')
    xgb = joblib.load('xgb_ensemble.sav')
    adab = joblib.load('adaboost_model.sav')
    pca = joblib.load('pca.sav')
    pipeline = joblib.load('pipeline.sav')
    return (pipeline, pca, log_reg, adab, xgb)

def predict(data):
    data.loc[data['RainToday'] == 'No', 'RainToday'] = 0.
    data.loc[data['RainToday'] == 'Yes', 'RainToday'] = 1.
    pipeline, pca, log_reg, adab, xgb = load()
    pipelined_data = pipeline.transform(data)
    pipelined_data = pca.transform(pipelined_data)
    pred_log = log_reg.predict(pipelined_data)
    pred_ad = adab.predict(pipelined_data)
    pred_xgb = xgb.predict(pipelined_data)
    return (pred_log[0], pred_ad[0], pred_xgb[0])
    
st.title("Modelos de Machine Learning para saber si llovió el día siguiente")
st.text("Universidad Autónoma de Chihuahua")
st.text("Facultad de Ingeniería")
st.text("Maestría en Ingeniería en Computación")
st.text("Machine Learning")
st.text("Proyecto Final Rain in Australia")
st.text("Gael Aristides Hinojos Ramírez")
st.text("384104")
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