import streamlit as st 
from src.components.model_selector import model_selection

def main():
  with st.sidebar:
    
    selected_model = model_selection()
     
  st.title("Welcome to the Model Playground")
  st.write("This is a simple model playground to test out different models")
  st.write("Select a model from the sidebar to get started")
  
  if selected_model is None:
    st.stop()
  
  st.session_state.model = selected_model['model']
  st.session_state.visualizer = selected_model['visualizer']()
  
  st.write(f"Selected model: {st.session_state.model.name}")
  st.write(f"Description: {st.session_state.model.description}")
  
  data_loading, inference, output = st.tabs(["Data Loading", "Inference", "Output"])
  with data_loading:
    st.session_state.visualizer.data_loading_step()
  with inference:
    st.session_state.visualizer.inference_step()
  with output:
    st.session_state.visualizer.output_step()
  
  
if __name__ == "__main__":
  main()