from src.models.supervised.base import SupervisedModel
from src.models.base import ModelNotFittedError

class OLS(SupervisedModel):
  name = "OLS"
  description = "Ordinary Least Squares"
  
  def __init__(self, intercept: bool = True):
    super().__init__()
    self._coefficients = None
    self._has_intercept = intercept
    
  def serialize(self):
    return {
      "name": self.name,
      "description": self.description
    }
  
  def load(self, serialized_model: dict):
    self.name = serialized_model["name"]
    self.description = serialized_model["description"]
    
  def fit(self, X, y):
    from statsmodels.api import OLS
    self.model = OLS(y, X).fit()
    self._is_fitted = True
    self._coefficients = self.model.params
    self.set_ivs(X)
    self.set_dvs(y)
  
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