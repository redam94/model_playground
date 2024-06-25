import streamlit as st
import pandas as pd

def dtype_selector(df):
    """
    Select the datatype of the data
    """
    dtype_df = pd.DataFrame(df.dtypes, columns=['dtype'])
    dtype_df['dtype'] = dtype_df['dtype'].astype(str)
    types = list(set(list(dtype_df['dtype'].unique())) | {"datetime", 'int', 'float', 'object'})
    
    dtypes = st.data_editor(
      dtype_df,
      column_config={
        'dtype': st.column_config.SelectboxColumn(
          "Selected Dtype",
          width="medium",
           options=types,
        )
      }
    )
    return dtypes
  
def apply_dtypes(dtypes, df):
  """Apply the selected datatypes to the dataframe"""
  
  df = df.copy()
  for col, dtype in dtypes.iterrows():
    if dtype['dtype'] == 'datetime':
      df[col] = pd.to_datetime(df[col])
    elif dtype['dtype'] == 'int':
      df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
    elif dtype['dtype'] == 'float':
      df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    elif dtype['dtype'] == 'object':
      df[col] = df[col].astype(str)
      
  return df
    
