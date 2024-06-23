"""Ordinary Least Squares (OLS) model for regression analysis"""

from src.models.supervised.base import SupervisedModel, SupervisedModelVisualizer
from src.models.base import ModelNotFittedError
from src.components.data_helpers.data_loader import dtype_selector, apply_dtypes

import io
from datetime import datetime
from uuid import uuid4

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class OLS(SupervisedModel):
  name = "OLS"
  description = "Ordinary Least Squares"
  
  def __init__(self, intercept: bool = True):
    super().__init__()
    self._has_intercept = intercept
    self.model = None
    
  def serialize(self):
    if not self._is_fitted:
      raise ModelNotFittedError()
    
    import pickle
    from io import BytesIO
    
    artifact_path = f"{self.name}_{uuid4()}_{datetime.now().isoformat()}.pkl"
    serialized_model = {
      "name": self.name,
      "description": self.description,
      "ivs": self._ivs,
      "dvs": self._dvs,
      "model": artifact_path
    }
    
    artifact = BytesIO()
    pickle.dump(self.model, artifact)

    return serialized_model, {'data': artifact, 'path': artifact_path}
  
  def load(self, fname: str|io.BytesIO):
    import pickle
    import json
    from zipfile import ZipFile
    
    with ZipFile(fname, 'r') as file:
      serialized_model = json.loads(file.read('model.json'))
      self.model = pickle.loads(file.read(serialized_model["model"]))
    self.name = serialized_model["name"]
    self.description = serialized_model["description"]
    self._ivs = serialized_model["ivs"]
    self._dvs = serialized_model["dvs"]
    self._is_fitted = True
    
  def fit(self, X, y):
    import statsmodels.api as sm
    if self._has_intercept:
      X = sm.add_constant(X)
    self.model = sm.OLS(y, X).fit()
    self._is_fitted = True
    
    self.set_ivs(X)
    self.set_dvs(y)
    
  @property
  def coefficients(self):
    if not self._is_fitted:
      raise ModelNotFittedError()
    return self.model.params
  
  def summary(self):
    if not self._is_fitted:
      raise ModelNotFittedError()
    return self.model.summary()
  
  def predict(self, X):
    if not self._is_fitted:
      raise ModelNotFittedError()
    
    y_pred = self.model.predict(X)
    return pd.DataFrame(y_pred, columns=self._dvs)
  
  def evaluate(self, X, y):
    if not self._is_fitted:
      raise Exception("Model not fitted")
    
    from sklearn.metrics import mean_squared_error
    y_pred = self.predict(X)
    return mean_squared_error(y, y_pred)
  

class OLSVisualizer(SupervisedModelVisualizer):
  """upload_data: Upload the data
process_data: Process the data
visualize_data: Visualize the data
transform_data: Transform the data
select_model_data: Select the data for the model
fit_model: Fit the model
output_step: Visualize the output step"""
  def __init__(self):
    super().__init__(OLS)
   
  def process_data(self):
    
    try:
      file_like = io.BytesIO(self.file_data)
      df = pd.read_csv(file_like)
    except Exception as e:
      st.error(f"Error loading file: {e}")
      return
    with st.expander("Data Processing Options"):
      with st.form(key='data_processing'):
        dtype_options, index_options = st.tabs(["Data Types", "Index"])
        options = list(df.columns)+["None"]
        with index_options:
          index = st.selectbox("Select the index variable", options=options, index=len(options)-1, key='index')
        
        with dtype_options:
          dtypes = dtype_selector(df)
          
        st.form_submit_button(label="Process Data")
    df = apply_dtypes(dtypes, df)
    processed_df = df
    if index != "None":
      processed_df = df.set_index(index)
      
    
    self.processed_data = processed_df

  def visualize_data(self):
    with st.expander("Data Summary"):
      data_head, data_summary, histogram, relationship = st.tabs(
        ["Data Preview", "Data Summary", "Histogram", "2-Way Relationship"]
      )
      
      with data_head:
        st.write(self.processed_data.head())
        
      with data_summary:
        st.write(self.processed_data.describe().transpose())
        
      with histogram:
        with st.form(key="value_deep_dive"):
          col = st.selectbox("Select a column", options=self.processed_data.columns)
          st.form_submit_button(label="Explore Variable")
        fig, ax = plt.subplots(figsize=(5, 3))
        try:
          self.processed_data[col].plot.hist(ax=ax)
          ax.set_title(f"Histogram of {col}")
          ax.set_xlabel(col)
        except TypeError:
          st.error("Cannot plot histogram")
        st.pyplot(fig)
        
      with relationship:
        with st.form(key="relationship"):
          y_var = st.selectbox("Select the dependent variable", options=self.processed_data.columns)
          x_var = st.selectbox("Select the independent variable", options=self.processed_data.columns, index=1)
          st.form_submit_button(label="Plot Relationship")
        fig, ax = plt.subplots(figsize=(16, 9))
        plt.scatter(self.processed_data[x_var], self.processed_data[y_var])
        ax.set_title(f"{y_var} vs {x_var}")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        st.pyplot(fig)
        
        
  def transform_data(self):
    pass
  
  def select_model_data(self):
    pass
  
  def fit_model(self):
    pass
  def output_step(self):
    pass