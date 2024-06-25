import streamlit as st 
from src.models.implemented import MODELS_IMPLEMENTED

def model_selection():
    
    st.title("Model Selection")
    
    problem_type = st.selectbox("Problem Type", list(MODELS_IMPLEMENTED.keys()))
    
    models = MODELS_IMPLEMENTED[problem_type]
    if "model" in models.keys():
        return models
  
    model_subtype = st.selectbox("Problem Subtype", list(models.keys()))
    
    models = models[model_subtype]
    if "model" in models.keys():
        return models
    
    data_type = st.selectbox("Data Type", list(models.keys()))
    
    models = models[data_type]
    if "model" in models.keys():
        return models
      
    model_type = st.selectbox("Model Type", list(models.keys()))
    
    models = models[model_type]
    if "model" in models.keys():
        return models
      
    model_name = st.selectbox("Model Name", list(models.keys()))
    
    models = models[model_name]
    if "model" in models.keys():
      return models
    