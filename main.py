import streamlit as st 
from src.models.supervised.regression.crosssection.linear_models.OLS import OLSVisualizer

model_visualizer = OLSVisualizer()
model_visualizer.data_loading_step()
