import pickle
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor




st.set_page_config(page_title='Social Profile LoanStatus Prediction App', page_icon="💸",
                               layout="wide", initial_sidebar_state='expanded')

pickle_in = open('XG_model.pkl', 'rb') 
XG_model = pickle.load(pickle_in)


# creating option list for dropdown menu

features = ['ListingNumber', 'ClosedDate', 'BorrowerAPR', 'BorrowerRate','DateCreditPulled', 'LoanMonthsSinceOrigination', 'LoanNumber',
       'LoanOriginationQuarter', 'LP_CustomerPayments','LP_CustomerPrincipalPayments']

st.markdown("<h1 style='color:white ;text-align: center;'>Social Profile LoanStatus Prediction App 💸 </h1>", unsafe_allow_html=True)


def prediction(num):
    if num ==0:
        return "Low Intensity"
    elif num==1:
        return "High Intensity"


def main():
    with st.form('prediction_form'):

        st.header("Predict the input for following features:") 
        st.header("XG Boost Regressor")
        st.header("![Alt Text](https://i.imgur.com/71KxTWy.jpg)")
        ListingNumber = st.selectbox( 'ListingNumber:', [1,4,6])
        
        ClosedDate = st.selectbox( 'ClosedDate:', [1,78,81,92,100,105])
        BorrowerAPR = st.selectbox('BorrowerAPR:', [1,747,902,1260,1644,3672])
        BorrowerRate = st.selectbox( 'BorrowerRate:', [1,1319,1508,1651,1905,3672])
        DateCreditPulled = st.selectbox('DateCreditPulled:', [1,4,6])
        LoanMonthsSinceOrigination = st.selectbox('LoanMonthsSinceOrigination:', [8,9,13,20,4336,4485,4899,5215,5865])
        LoanNumber = st.selectbox('LoanNumber:', [1,4,6])
        LoanOriginationQuarter = st.selectbox('LoanOriginationQuarter:', [13,585,1243,1270,1600,2403,3074,3913,4424,5061,14450])
        LP_CustomerPayments = st.selectbox('LP_CustomerPayments:', [1,82,83,119,6208])
        LP_CustomerPrincipalPayments = st.selectbox('LP_CustomerPrincipalPayments:', [1,1938,2042,2274,2595,6308])
        button = st.form_submit_button('Predict')
       
    if button:

        data= np.array([ListingNumber, ClosedDate, BorrowerAPR, BorrowerRate, DateCreditPulled,LoanMonthsSinceOrigination,LoanNumber,
                        LoanOriginationQuarter,LP_CustomerPayments,LP_CustomerPrincipalPayments]).reshape(1, -1)
        
        prediction = XG_model.predict(data)
            
        st.write(f"The predicted LoanStatus is:  {prediction}")   
            
        


if __name__ == '__main__':
    main()        