import streamlit as st
import pickle 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import random 
from PIL import Image, ImageOps 
import warnings 
from tensorflow.keras.preprocessing import image 
from  streamlit_image_select import image_select 
warnings.filterwarnings('ignore') 


model = tf.keras.models.load_model('X-ray_classifier.h5')
st.set_page_config(
    page_title= 'Covid 19 Detection by X-ray Images', 
    page_icon='😷' , 
    initial_sidebar_state = 'auto' 
)

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.title('😷 Covid-19 Detection from X-Ray 🩻 Images...')


def predict_result(prediction) : 
    class_name = ['Covid-19 Negetive' , 'Covid-19 Positive'] 
    string = 'from Uploaded X-ray Image we estimate that you are :'+ class_name[prediction] 
    if prediction == 0 :  
        st.balloons() 
        st.info(string) 
    
    elif prediction == 1 : 
        st.info(string)


def prediction_cls(prediction) : 

    for key , clss in class_name.items() : 
        if np.argmax(prediction) == clss : 
            
            return key 


with st.sidebar: 
    st.image('covid.jpeg')
    st.title('Covid🦠19 Detector🔍') 
    st.subheader("Instantly discern COVID-19 status from uploaded images! 📸 Reliable classification of positive or negative cases aids swift identification.")
    st.write('Covid-19 Detection by uploading Image')
    st.link_button('Github Source-Code' , url= 'https://www.kaggle.com/work/overview') 


file = st.file_uploader('Upload X-ray Image',type = ['jpeg','jpg','png']) 

def import_and_predict(data , model) : 
    img = image.load_img(data , target_size = (200,200,3)) 
    img = image.img_to_array(img) 
    img = np.expand_dims(img , axis= 0) 
    prediction = model.predict(img) 
    prediction = np.argmax(prediction) 
    return prediction

if file is None : 
    st.text('Please Upload Image File') 
    st.subheader('Sample Images') 
    images  = ['pos.jpeg','negetive.jpeg']
    selected_img = image_select('1.Positive , 2.Negetive',images , key ='click_images')
    prediction = import_and_predict(selected_img , model)
    if st.button('Predict') : 
        st.image(selected_img) 
        predict_result(prediction) 


else : 
    prediction = import_and_predict(file , model) 
    st.image(file)
    if st.button('Predict') : 
        predict_result(prediction)
