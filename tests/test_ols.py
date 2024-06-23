import sys
sys.path.append('.')
sys.path.append("..")

from src.models.supervised.regression.crosssection.linear_models.OLS import OLS
from src.models.base import ModelNotFittedError
import numpy as np
import pandas as pd
import unittest

class TestOLS(unittest.TestCase):
  def setUp(self):
    self.fitted_model = OLS()
    X = pd.DataFrame(np.random.randn(100, 2), columns=['x1', 'x2'])
    y = pd.Series(np.random.randn(100), name='y')
    self.fitted_model.fit(X, y)
    self.unfitted_model = OLS()
    
  def test_fit(self):
    
    self.assertFalse(self.unfitted_model._is_fitted)
    self.assertTrue(self.unfitted_model.model is None)
    
    self.assertTrue(self.fitted_model._is_fitted)
    self.assertTrue(self.fitted_model.model is not None)
    
  def test_model(self):
    X = pd.DataFrame(np.random.randn(100, 2), columns=['x1', 'x2'])
    y = pd.Series(np.random.randn(100), name='y')
    model = OLS()
    model.fit(X, y)
    import statsmodels.api as sm
    X = sm.add_constant(X)
    sm_model = sm.OLS(y, X).fit()
    self.assertTrue(np.allclose(model.model.params, sm_model.params))
    
  def test_serialize(self):
    
    
    self.assertRaises(ModelNotFittedError, self.unfitted_model.serialize)
    serialized_model, artifact = self.fitted_model.serialize()
    self.assertTrue(serialized_model['name'] == 'OLS')
    self.assertTrue(serialized_model['description'] == 'Ordinary Least Squares')
    self.assertTrue(serialized_model['ivs'] == ['const', 'x1', 'x2'])
    self.assertTrue(serialized_model['dvs'] == ['y'])
    self.assertTrue(artifact['path'].startswith('OLS_'))
    
  def test_summary(self):
    
    self.assertRaises(ModelNotFittedError, self.unfitted_model.summary)
    
    self.assertTrue(self.fitted_model.summary() is not None)
    print(self.fitted_model.summary())
if __name__ == '__main__':
  unittest.main()