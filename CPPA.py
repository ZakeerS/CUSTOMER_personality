#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import streamlit as st
import pickle



def segment_customers(prediction):
    pred_1 = ''
    if prediction == 0:
        pred_1 = 'Cluster 0 - Good Customer'
    elif prediction == 1:
        pred_1 = 'Cluster 1 - Below Average Customer'
    elif prediction == 2:
        pred_1 = 'Cluster 2 - Elite Customer'
    elif prediction == 3:
        pred_1 = 'Cluster 3 - Average Customer'
    else:
        pred_1 = 'Cluster 4 - Best Customer'
    return pred_1


def main():
    st.title("CUSTOMER PERSONALITY ANALYSIS")
    st.sidebar.header("USER INPUT PARAMETERS")
    Income = st.sidebar.text_input("Type In The Household Income")
    Average_Spent = st.sidebar.text_input("Type in Average Spent")
    Kids = st.sidebar.radio("Select Number Of Kids In Household", ('0', '1', '2', '3', '4'))
    Customer_Age = st.sidebar.slider("Select Age", 18, 90)
    Education_Level = st.sidebar.radio("Select Education(0-High, 1-Medium,2-Low)", ("0", "1", "2"))
    Marital_Status = st.sidebar.radio("Marital status(0-Relationship, 1-Single)",("0","1"))
    NumDealsPurchases = st.sidebar.text_input("Deals purchased with discount")
    Complain = st.sidebar.radio("Complain(0-No, 1-Yes)",("0","1"))
    Response = st.sidebar.radio("Response(0-No, 1-Yes)",("0","1"))
    TotalAcceptedCmp = st.sidebar.text_input("Total campaigns accepted")
    avg_web_visits = st.sidebar.text_input("Average web visits")
    NumTotalPurchases = st.sidebar.text_input("Total number of purchases")
    Is_Parent = st.sidebar.radio("Is the customer a parent(0-No, 1-Yes)",("0","1"))
    Customer_For = st.sidebar.text_input("For how many days the customer is enrolled:")

    result = ""
    data = {
        'Income': [Income],
        "Average_Spent": [Average_Spent],
        'Kids': [Kids],
        'Customer_Age': [Customer_Age],
        "Education_Level": [Education_Level],
        "Marital_Status":[Marital_Status],
        "NumDealsPurchases":[NumDealsPurchases],
        "Complain":[Complain],
        "Response":[Response],
        "TotalAcceptedCmp":[TotalAcceptedCmp],
        "avg_web_visits":[avg_web_visits],
        "NumTotalPurchases":[NumTotalPurchases],
        "Is_Parent":[Is_Parent],
        "Customer_For":[Customer_For]
    }
    features = pd.DataFrame(data)
    st.write(features)

    if st.button("Segment Customer"):
        # Load the model from file
        loaded_model = pickle.load(open('CPA_final.sav', 'rb'))

        # Make predictions
        prediction = loaded_model.predict(features.values)
        prediction_proba = loaded_model.predict_proba(features.values)

        result = segment_customers(prediction[0])
        st.success(result)

        st.write("Predicted class label:", prediction[0])
        st.write("Predicted probabilities:")
        st.write(prediction_proba)

if __name__ == '__main__':
    main()





