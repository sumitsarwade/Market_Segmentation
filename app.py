
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
 
#load the model from disk
loaded_model = pickle.load(open(r'E://MarketSegmentation-main//final_model.sav', 'rb'))


df = pd.read_csv(r"E://MarketSegmentation-main//Clustered_Customer_Data.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Market Segmentation")

left_column, right_column = st.columns(2)

with left_column:
 with st.form("my_form"):
  

        BALANCE= st.slider("balance Income",0,200000)
        balance_frequency=st.number_input(label='Balance Frequency',step=0.001,format="%.6f")
        PURCHASES= st.slider("PURCHASES",0,10000)
        oneoff_purchases=st.number_input(label='OneOff_Purchases',step=0.01,format="%.2f")
        installments_purchases=st.number_input(label='Installments Purchases',step=0.01,format="%.2f")
        CASH_ADVANCE= st.slider("CASH_ADVANCE",0,10000)
        purchases_frequency=st.number_input(label='Purchases Frequency',step=0.01,format="%.6f")
        oneoff_purchases_frequency=st.number_input(label='OneOff Purchases Frequency',step=0.1,format="%.6f")
        purchases_installment_frequency=st.number_input(label='Purchases Installments Freqency',step=0.1,format="%.6f")
        CASH_ADVANCE_FREQUENCY= st.slider("CASH_ADVANCE_FREQUENCY",0.0,0.90000)
        CASH_ADVANCE_TRX= st.slider("CASH_ADVANCE_TRX",0,50)
        PURCHASES_TRX= st.slider("PURCHASES_TRX",0,100)
        CREDIT_LIMIT= st.slider("CREDIT_LIMIT",0,50000)
        PAYMENTS= st.slider("PAYMENTS",0,20000)
        MINIMUM_PAYMENTS= st.slider("MINIMUM_PAYMENTS",0,20000)
        prc_full_payment=st.number_input(label='PRC Full Payment',step=0.01,format="%.6f")
        tenure=st.number_input(label='Tenure',step=1)

        data=[[BALANCE,balance_frequency,PURCHASES,oneoff_purchases,installments_purchases,CASH_ADVANCE,purchases_installment_frequency,oneoff_purchases_frequency,purchases_installment_frequency,CASH_ADVANCE_FREQUENCY,CASH_ADVANCE_TRX,PURCHASES_TRX,CREDIT_LIMIT,PAYMENTS,MINIMUM_PAYMENTS,prc_full_payment,tenure]]

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

    