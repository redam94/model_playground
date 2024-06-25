from src.models import supervised, unsupervised

MODELS_IMPLEMENTED = {
  "Supervised": {
    "Regression": {
      "Cross Sectional": {
        "Linear": {
          "OLS": {
            "model": supervised.regression.crosssection.linear_models.OLS,
            "visualizer": supervised.regression.crosssection.linear_models.OLSVisualizer,
          }
        }
      },
    }
  },
}
