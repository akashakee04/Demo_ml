import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
st.title("Medical Diagnostic web app")

#Step 1 : Load the model

model = open('rfc.pickle', 'rb')
rfc_model = pickle.load(model)
model.close()

#step 2 : Create a Ui for front end user
pregs = st.slider('Pregnancies',0,20, step = 1)
Glucose = st.slider('Glucose',40,200,40)
bp = st.slider('BloodPressure',20,240,24)
skin = st.slider('SkinThickness',5, 100, 5)
Insulin = st.slider('Insulin',14,900,14)
BMI = st.slider('BMI',15,70,15)
dpf = st.slider('DiabetesPedigreeFunction',0.05, 2.50,0.05)
age = st.slider('Age', 21,90,21)

#step3: Change user input to model input data

data = {['Pregnancies': pregs,
         'Glucose': Glucose,
         'BloodPressure': bp,
         'SkinThickness': skin,
         'Insulin': Insulin,
       'BMI':BMI,
         'DiabetesPedigreeFunction':dpf,
         'Age':age]}

input_data = pd.DataFrame([data])

#step 4: get predictions and print the result
predictions = rfc_model.predict(input_data)[0]
st.write(predictions)
if st.button("predict"):
    if predictions ==0:
        st.success("diabetes free")
    if predictions ==1:
        st.error('Has diabetes')
        


