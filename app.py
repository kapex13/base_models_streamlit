import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

subprocess.run([sys.executable, "-m", "pip", "install", "joblib"])


import streamlit as st
import pandas as pd
import joblib

# Загрузка модели
model = joblib.load('best_model.pkl')

st.title('Прогноз сердечных заболеваний')

# Форма ввода данных
with st.form('patient_form'):
    age = st.number_input('Возраст', min_value=0, max_value=120)
    sex = st.selectbox('Пол', ['M', 'F'])
    chest_pain = st.selectbox('Тип боли в груди', ['TA', 'ATA', 'NAP', 'ASY'])
    resting_bp = st.number_input('Давление в покое', min_value=0)
    cholesterol = st.number_input('Холестерин', min_value=0)
    fasting_bs = st.selectbox('Сахар натощак >120', [0, 1])
    resting_ecg = st.selectbox('ЭКГ в покое', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Макс. пульс', min_value=60, max_value=220)
    exercise_angina = st.selectbox('Стенокардия при нагрузке', ['N', 'Y'])
    oldpeak = st.number_input('Oldpeak', min_value=0.0)
    st_slope = st.selectbox('Наклон ST', ['Up', 'Flat', 'Down'])
    
    submitted = st.form_submit_button('Прогноз')

# Прогнозирование
if submitted:
    data = [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, 
             resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]]
    cols = ['Age','Sex','ChestPainType','RestingBP','Cholesterol',
            'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    
    df = pd.DataFrame(data, columns=cols)
    proba = model.predict_proba(df)[0][1]
    
    st.success(f'Вероятность заболевания: {proba*100:.1f}%')
    st.write('Высокий риск' if proba > 0.5 else 'Низкий риск')