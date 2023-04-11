# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
st.set_page_config(page_title='BTCS_G01', page_icon="https://cdn-icons-png.flaticon.com/128/5017/5017903.png",layout="wide")

#loading the logistic regression model
whole = pickle.load(open("C:/Users/dhara/Desktop/Breast Cancer Project_G01/model_lr.sav",'rb'))
worst = pickle.load(open("C:/Users/dhara/Desktop/Breast Cancer Project_G01/mrf.sav",'rb'))

with st.sidebar:
    selected = option_menu('Breast Cancer Prediction Models',
                           
                           ['Home',
                            'All attributes',
                           'Worst attributes only'],
                           menu_icon="cast",
                           icons=['house','activity','activity'],
                           default_index=0)
if(selected=="Home"):
    st.markdown("<h1 style='text-align: center; '>BTCS_G01 - Breast Cancer Classification</h1>", unsafe_allow_html=True)
    st.divider()
    col1,col2= st.columns(2)
    with col1:
        image = Image.open(r'C:\Users\dhara\Desktop\Breast Cancer Project_G01\research_papers\logo.png')
        st.image(image, width=350)
    with col2:
        st.write('Hello, *World!* :sunglasses:  \n\n This is our ML project-Breast Cancer Classification.  \n\nAfter much research we have a Logistic regression model which uses all the attributes of the Wisconsin dataset and we have a Random Forest model that only utilizes the worst attributes which came out to be predicting as good as the model that uses all the attributes according to our study.  \n\nThis deployment is to check our results and do further resesarch on the topic.	:purple_heart:')
    
    
if(selected=='All attributes'):
    st.markdown("<h1 style='text-align: center; '>Logistic Regression</h1>", unsafe_allow_html=True)
    
    st.divider()
    st.text('Enter the test values below:')
    col1,col2,col3,col4 = st.columns(4)
    
    with col1:
        radius_mean =st.number_input('radius_mean')
    with col2:    
        texture_mean =st.number_input('texture_mean')  
    with col3:
        perimeter_mean =st.number_input('perimeter_mean')
    with col1:
        area_mean =st.number_input('area_mean')
    with col4:
        smoothness_mean =st.number_input('smoothness_mean')
    with col1:
        compactness_mean =st.number_input('compactness_mean')
    with col2:
        concavity_mean=st.number_input('concavity_mean')    
    with col3:
        concave_points_mean =st.number_input('concave points_mean') 
    with col4:
        symmetry_mean =st.number_input('symmetry_mean')   
    with col1:
        fractal_dimension_mean =st.number_input('fractal_dimension_mean')
        
    with col2:
        radius_se =st.number_input('radius_se')
    with col3:
        texture_se =st.number_input('texture_se') 
    with col4:
        perimeter_se =st.number_input('perimeter_se') 
    with col1:
        area_se =st.number_input('area_se')    
    with col2:
        smoothness_se =st.number_input('smoothness_se')
    with col3:
        compactness_se =st.number_input('compactness_se')
    with col4:
        concavity_se =st.number_input('concavity_se') 
    with col1:
        concave_points_se =st.number_input('concave points_se')
    with col2:
        symmetry_se =st.number_input('symmetry_se')    
    with col3:
        fractal_dimension_se =st.number_input('fractal_dimension_se')
        
    with col4:
        radius_worst =st.number_input('radius_worst')
    with col1:
        texture_worst =st.number_input('texture_worst') 
    with col2:
        perimeter_worst =st.number_input('perimeter_worst')
    with col3:
        area_worst =st.number_input('area_worst')    
    with col4:
        smoothness_worst=st.number_input('smoothness_worst')  
    with col1:
        compactness_worst=st.number_input('compactness_worst')
    with col2:
        concavity_worst =st.number_input('concavity_worst')    
    with col3:
        concave_points_worst =st.number_input('concave points_worst')  
    with col4:
        symmetry_worst =st.number_input('symmetry_worst') 
    with col2:
        fractal_dimension_worst=st.number_input('fractal_dimension_worst')
    
    #code for prediction
    diag=''
    
    #button for prediction
    if st.button('Test Result'):
        pred = whole.predict([[ radius_mean,
                                texture_mean,
                                perimeter_mean,
                                area_mean,
                                smoothness_mean,
                                compactness_mean,
                                concavity_mean,
                                concave_points_mean,
                                symmetry_mean,
                                fractal_dimension_mean,
                                radius_se,
                                texture_se,
                                perimeter_se,
                                area_se,
                                smoothness_se,
                                compactness_se,
                                concavity_se,
                                concave_points_se,
                                symmetry_se,
                                fractal_dimension_se,
                                radius_worst,
                                texture_worst,
                                perimeter_worst,
                                area_worst,
                                smoothness_worst,
                                compactness_worst,
                                concavity_worst,
                                concave_points_worst,
                                symmetry_worst,
                                fractal_dimension_worst]])
        if(pred[0]==1):
            diag='Tumor is malignant'
        else:
            diag='Tumor is benign'
    st.success(diag)
    
if(selected=='Worst attributes only'):
    st.markdown("<h1 style='text-align: center; '>Random forest</h1>", unsafe_allow_html=True)
    
    st.divider()
    st.text('Enter the test values below:')
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        radius_worst =st.number_input('radius_worst')
    with col2:
        texture_worst =st.number_input('texture_worst') 
    with col3:
        perimeter_worst =st.number_input('perimeter_worst')
    with col4:
        area_worst =st.number_input('area_worst')    
    with col1:
        smoothness_worst=st.number_input('smoothness_worst')  
    with col2:
        compactness_worst=st.number_input('compactness_worst')
    with col3:
        concavity_worst =st.number_input('concavity_worst')    
    with col4:
        concave_points_worst =st.number_input('concave points_worst')  
    with col1:
        symmetry_worst =st.number_input('symmetry_worst') 
    with col2:
        fractal_dimension_worst=st.number_input('fractal_dimension_worst')
    
    #code for prediction
    diag=''
    
    #button for prediction
    if st.button('Test Result'):
        pred = worst.predict([[ radius_worst,
                                texture_worst,
                                perimeter_worst,
                                area_worst,
                                smoothness_worst,
                                compactness_worst,
                                concavity_worst,
                                concave_points_worst,
                                symmetry_worst,
                                fractal_dimension_worst]])
        if(pred[0]==1):
            diag='Tumor is malignant'
        else:
            diag='Tumor is benign'
    st.success(diag)