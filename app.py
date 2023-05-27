from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from pycaret.classification import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    df = df.dropna()


with st.sidebar:
    st.title("Automate ML")
    choice = st.radio("Select Options : ",['Upload File','Dataset Info','Analysis','ML Model','Download'])

if choice == "Upload File":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Dataset Info": 
    st.title("Dataset Info")
    profile_df = df.profile_report()
    if st.button("Show Data Info"):
        st_profile_report(profile_df)

    
if choice == "Analysis": 
    st.title("Exploratory Data Analysis")
    report = sv.analyze([df,'EDA'])
    if st.button("Show Analysis"):
        report_html = report.show_html()

if choice == "ML Model":
    ml = st.selectbox("Select ",options=['Regression','Classification'])
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        if ml == 'Regression':
            setup(df, target=chosen_target, )
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
        else:
            setup(df, target=chosen_target,)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models(sort='AUC')
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
            
            
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
    