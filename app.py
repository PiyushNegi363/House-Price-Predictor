import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load the trained model
model = pickle.load(open('RidgeModel.pkl','rb'))
df = pickle.load(open('house_data.pkl','rb'))

st.title("House Price Predictor")

location = st.selectbox('Select the Location',sorted(df['location'].unique()))

bhk = st.selectbox('Select the BHK',sorted(df['BHK'].unique()))

sq_feet = st.selectbox("Select the Total Square Feet",df['total_sqft'].unique())

bath = st.selectbox('Select the Bath',sorted(df['bath'].unique()))

if st.button("Predict Price"):
    st.header("The predicted price of the house is")
    query = pd.DataFrame([[location,sq_feet,bath,bhk]],columns=['location','total_sqft','bath','BHK'])
    result = model.predict(query)[0]
    st.header(f"₹ {result:,.0f}0000")
    
