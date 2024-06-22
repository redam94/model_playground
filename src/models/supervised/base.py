import pandas as pd
from src.models.base import BaseModel
import abc

class SupervisedModel(BaseModel):
  def __init__(self):
    self._ivs = None
    self._dvs = None
    self._is_fitted = False
    
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
