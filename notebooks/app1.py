
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
 
#load the model from disk
loaded_model = pickle.load(open(r'E://Market_Segmentation//notebooks//final_model.sav', 'rb'))


df = pd.read_csv(r"E://Market_Segmentation//notebooks//Clustered_Customer_Data.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Market Segmentation")

left_column, right_column = st.columns(2)

with left_column:
 with st.form("my_form"):
  

        BALANCE= st.slider("balance Income",0,20000)
        BALANCE_FREQUENCY= st.slider("BALANCE_FREQUENCY",0.0,1.0)
        PURCHASES= st.slider("PURCHASES",0,10000)
        ONEOFF_PURCHASES= st.slider("ONEOFF_PURCHASES",0.0,10000.0)
        INSTALLMENTS_PURCHASES= st.slider("INSTALLMENTS_PURCHASES",0,10000)
        CASH_ADVANCE= st.slider("CASH_ADVANCE",0,10000)
        PURCHASES_FREQUENCY= st.slider("PURCHASES_FREQUENCY",0,10000)
        ONEOFF_PURCHASES_FREQUENCY= st.slider("ONEOFF_PURCHASES_FREQUENCY",0.0,0.1000)
        PURCHASES_INSTALLMENTS_FREQUENCY= st.slider("PURCHASES_INSTALLMENTS_FREQUENCY",0.0,0.90000)
        CASH_ADVANCE_FREQUENCY= st.slider("CASH_ADVANCE_FREQUENCY",0.0,0.90000)
        CASH_ADVANCE_TRX= st.slider("CASH_ADVANCE_TRX",0.0,1.0)
        PURCHASES_TRX= st.slider("PURCHASES_TRX",0,200)
        CREDIT_LIMIT= st.slider("CREDIT_LIMIT",0,10000)
        PAYMENTS= st.slider("PAYMENTS",0,20000)
        MINIMUM_PAYMENTS= st.slider("MINIMUM_PAYMENTS",0,20000)
        PRC_FULL_PAYMENT= st.slider("PRC_FULL_PAYMENT",0.0,0.1)
        TENURE= st.slider("TENURE",0,20)
        data=[[BALANCE,BALANCE_FREQUENCY,PURCHASES,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE,PURCHASES_FREQUENCY,ONEOFF_PURCHASES_FREQUENCY,PURCHASES_INSTALLMENTS_FREQUENCY,CASH_ADVANCE_FREQUENCY,CASH_ADVANCE_TRX,PURCHASES_TRX,CREDIT_LIMIT,PAYMENTS,MINIMUM_PAYMENTS,PRC_FULL_PAYMENT,TENURE]]

        submitted = st.form_submit_button("Submit")

        if submitted:
            clust=loaded_model.predict(data)[0]
            st.write('customer Belongs to Cluster',clust)

if submitted:               
    with right_column:
        cluster_df1=df[df['Cluster']==clust]
        fig, ax = plt.subplots()
        
        plt.title("Histogram")

        for c in cluster_df1.drop(['Cluster'],axis=1):
            grid= sns.FacetGrid(cluster_df1, col='Cluster')
            grid= grid.map(plt.hist, c)
            st.pyplot(grid)

    