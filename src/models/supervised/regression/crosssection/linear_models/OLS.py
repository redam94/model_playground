from src.models.supervised.base import SupervisedModel
from src.models.base import ModelNotFittedError

import pandas as pd
import io
from datetime import datetime
from uuid import uuid4
import os

class OLS(SupervisedModel):
  name = "OLS"
  description = "Ordinary Least Squares"
  
  def __init__(self, intercept: bool = True):
    super().__init__()
    self._has_intercept = intercept
    
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
  
  def load(self, fname: str):
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