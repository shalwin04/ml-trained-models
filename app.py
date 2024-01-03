import streamlit as st
import pandas as pd 
import os 
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#ML
from pycaret.classification import setup,compare_models,pull,save_model

with st.sidebar:

    st.title("Automatatic ml model")
    choice = st.radio("Navigation",["upload","profiling","ml","download"])
    st.info("This app allows you to build autmated ml pipeline")

if os.path.exists("sourcedata.csv"):
    df= pd.read_csv("sourcedata.csv",index_col=None)
else:
    pass

if choice=="upload":
    st.title("* Upload the Dataset for Modelling *")
    file = st.file_uploader("Upload your dataset")
    if(file):
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv" , index=None)
        st.dataframe(df)
if choice == "profiling":
    st.title("Automated EDA")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ml":
    st.title("Machine Learnig models")
    target = st.selectbox("Select Your target", df.columns)
    setup(df,target=target)
    setup_df=pull()
    st.info("This is the Ml Experiment settings")
    st.dataframe(setup_df)
    best_model = compare_models()
    compare_df = pull()
    st.info("This is the ML model")
    st.dataframe(compare_df)
    best_model
