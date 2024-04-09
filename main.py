import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings

warnings.filterwarnings("ignore")

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def main():
    st.title('Breast Cancer Prediction')
    st.write('Enter the values for the following features to predict breast cancer:')

    mean_radius = st.slider('Mean Radius', 0.0, 30.0, 15.0)
    mean_texture = st.slider('Mean Texture', 0.0, 40.0, 20.0)
    mean_perimeter = st.slider('Mean Perimeter', 0.0, 200.0, 100.0)
    mean_area = st.slider('Mean Area', 0.0, 2500.0, 1250.0)
    mean_smoothness = st.slider('Mean Smoothness', 0.0, 0.2, 0.1)
    mean_compactness = st.slider('Mean Compactness', 0.0, 0.4, 0.2)
    mean_concavity = st.slider('Mean Concavity', 0.0, 0.5, 0.25)
    mean_concave_points = st.slider('Mean Concave Points', 0.0, 0.2, 0.1)
    mean_symmetry = st.slider('Mean Symmetry', 0.0, 0.5, 0.25)
    mean_fractal_dimension = st.slider('Mean Fractal Dimension', 0.0, 0.1, 0.05)
    radius_error = st.slider('radius error', 0.0, 0.1, 0.2)
    texture_error = st.slider('texture error', 0.0, 0.1, 0.1)
    perimeter_error = st.slider('perimeter error', 0.0, 0.1, 0.05)
    area_error = st.slider('area error', 0.0, 0.1, 0.2)
    smoothness_error = st.slider('smoothness error', 0.0, 0.1, 0.1)
    compactness_error = st.slider('compactness error', 0.0, 0.1, 0.05)
    concavity_error = st.slider('concavity error', 0.0, 0.1, 0.25)
    concave_points_error = st.slider('concave points error', 0.0, 0.1, 0.2)
    symmetry_error = st.slider('symmetry error', 0.0, 0.1, 0.1) 
    fractal_dimension_error= st.slider('fractal dimension error', 0.0, 0.1, 0.05)  
    worst_radius = st.slider('worst radius', 0.0, 0.1, 0.2) 
    worst_texture = st.slider('worst texture', 0.0, 0.1, 0.25) 
    worst_perimeter = st.slider('worst perimeter', 0.0, 0.1, 0.05) 
    worst_area = st.slider('worst area', 0.0, 0.1, 0.1) 
    worst_smoothness = st.slider('worst smoothness', 0.0, 0.1, 0.2) 
    worst_compactness = st.slider('worst compactness', 0.0, 0.1, 0.05) 
    worst_concavity = st.slider('worst concavity', 0.0, 0.1, 0.25) 
    worst_concave_points = st.slider('worst concave points', 0.0, 0.1, 0.1) 
    worst_symmetry = st.slider('worst symmetry', 0.0, 0.1, 0.2) 
    worst_fractal_dimension = st.slider('worst fractal dimension', 0.0, 0.1, 0.05) 


    
    features_value = np.array([mean_radius, mean_texture, mean_perimeter, mean_area,
                                mean_smoothness, mean_compactness, mean_concavity,
                                mean_concave_points, mean_symmetry, mean_fractal_dimension,radius_error,
                                texture_error, perimeter_error, area_error,
                                smoothness_error,compactness_error,concavity_error,
                                concave_points_error,symmetry_error,fractal_dimension_error,
                                worst_radius,worst_texture,worst_perimeter,worst_area,
                                worst_smoothness,worst_compactness,worst_concavity,
                                worst_concave_points,worst_symmetry,worst_fractal_dimension]).reshape(1, -1)

    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                      'mean smoothness', 'mean compactness', 'mean concavity',
                      'mean concave points', 'mean symmetry', 'mean fractal dimension',
                      'radius error', 'texture error', 'perimeter error', 'area error',
                      'smoothness error', 'compactness error', 'concavity error',
                      'concave points error', 'symmetry error', 'fractal dimension error',
                      'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                      'worst smoothness', 'worst compactness', 'worst concavity',
                      'worst concave points', 'worst symmetry', 'worst fractal dimension']

    df = pd.DataFrame(features_value, columns=features_name)

    st.subheader('Prediction:')

    if st.button('predict'):
        # Preprocess the uploaded image and predict the class
     prediction = model.predict(df)
     st.success(f'Prediction: {str(prediction)}')
     if prediction == 0:
        res_val = "** breast cancer **"
     else:
        res_val = "no breast cancer"
     st.success(res_val)


    # if prediction == 0:
    #     res_val = "** breast cancer **"
    # else:
    #     res_val = "no breast cancer"

    # st.write('Patient has', res_val)

    

if __name__ == "__main__":
    main()
