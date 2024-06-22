"""Contains Base Model class for all models to inherit from and classes helpful to all models"""

from datetime import datetime
from uuid import uuid4
import abc

class ModelNotFittedError(AttributeError):
  """Model is not fitted error"""
  def __init__(self, message="Model is not fitted"):
    self.message = message
    super().__init__(self.message)

class BaseModel(abc.ABC):
  """Base Model class for all models to inherit from"""
  name: str
  description: str
  
  @abc.abstractmethod
  def serialize(self):
    """Serialize the model to a dictionary"""
    pass
  
  def save(self, filename: str = None):
    """Save the model to a file"""
    if filename is None:
      filename = f"{self.name}_{datetime.now().isoformat()}_{uuid4()}.model"
    
    with open(filename, "w") as f:
      f.write(self.serialize())
  
  @abc.abstractmethod
  def load(self, serialized_model: dict):
    """Load the model from serialized data"""
    pass
  
  