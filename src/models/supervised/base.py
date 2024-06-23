"""This module defines the base classes for supervised learning models and their visualizers"""
from src.models.base import BaseModel, BaseModelVisualizer

import abc

import streamlit as st
import pandas as pd

class SupervisedModel(BaseModel):
  """Base class for all supervised learning models"""
  def __init__(self):
    self._ivs = None # Independent variables
    self._dvs = None # Dependent variables
    self._is_fitted = False # Whether the model is fitted
    
  @abc.abstractmethod
  def fit(self, X, y, *args, **kwargs):
    """Train the model"""
    pass
  
  @abc.abstractmethod
  def predict(self, X):
    """Predict the target variable"""
    pass
  
  @abc.abstractmethod
  def evaluate(self, X, y):
    """Evaluate the model"""
    pass
  
  def set_ivs(self, ivs: pd.DataFrame):
    self._ivs = list(ivs.columns)
  
  def set_dvs(self, dvs: pd.DataFrame | pd.Series):
    if isinstance(dvs, pd.Series):
      self._dvs = [dvs.name]
    else:
      self._dvs = list(dvs.columns)

class SupervisedModelVisualizer(BaseModelVisualizer):
  """ Base Class for visualizing supervised models
  Attributes:
  model: SupervisedModel: The supervised model to visualize
  file_data: bytes: The uploaded file data
  processed_data: pd.DataFrame: The processed data
  transformed_data: pd.DataFrame: The transformed data
  model_data: pd.DataFrame: The data to use for the model
  Methods:
  
  upload_data: Upload the data
  process_data: Process the data
  visualize_data: Visualize the data
  transform_data: Transform the data
  select_model_data: Select the data for the model
  fit_model: Fit the model
  output_step: Visualize the output step
  
  data_loading_step: Visualize the data loading step
  inference_step: Visualize the inference step
  
  """
  def __init__(self, model: SupervisedModel):
    self.model = model
    self.file_data = None
    self.processed_data = None
    self.transformed_data = None
    self.model_data = None
  
  def upload_data(self):
    with st.expander("Upload Data"):
      file = st.file_uploader("Upload a your data file")
      if file is not None:
        self.file_data = file.getvalue()
  
  @abc.abstractmethod
  def process_data(self):
    """Process the data"""
    pass
  
  @abc.abstractmethod
  def visualize_data(self):
    """Visualize the data"""
    pass
  
  @abc.abstractmethod
  def transform_data(self):
    """Transform the data"""
    pass
  
  @abc.abstractmethod
  def select_model_data(self):
    """Select the data for the model"""
    pass
  
  def data_loading_step(self):
    self.upload_data()
    
    if self.file_data is None:
      st.stop()
    
    self.process_data()
    
    if self.processed_data is None:
      st.stop()
      
    self.transformed_data = self.processed_data
    self.transform_data()
    
    if self.transformed_data is None:
      st.stop()
    
    self.visualize_data()
  
  @abc.abstractmethod
  def fit_model(self):
    """Fit the model"""
    pass

  def inference_step(self):
    """Visualize the inference step"""
    self.select_model_data()
    if self.model_data is None:
      st.stop()
      
    self.fit_model()
    
    if self.model._is_fitted:
      st.success("Model fitted successfully")
    if not self.model._is_fitted:
      st.stop()
  
  @abc.abstractmethod
  def output_step(self):
    """Visualize the output step"""
    pass