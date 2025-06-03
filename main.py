import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import base64
import requests
import pandas
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from streamlit_option_menu import option_menu
st.set_page_config("Agro",page_icon=":seedling:",layout="wide")
import joblib
import process

with open('agro12.jpg','rb') as f:
    data=f.read()
imgs= base64.b64encode(data).decode()

css=f"""
    <style>
    [data-testid="stAppViewContainer"]{{
        background-image: url('data:image/png;base64,{imgs}');
        background-size:cover
    }}
    [data-testid="stSidebar"]{{
        background-color:#becc58
    }}
   </style>
"""
st.markdown(css,unsafe_allow_html=True)

with st.sidebar:
    st.header("AgroVista")
    selected=option_menu(
        menu_title="Dashbord",
        options=["Home","Plant Diseases","Plant Identification","Crop Yield","About"],
        icons=["house-door","flower1","tree","info-circle"],
        menu_icon=":arrow_double_down:",
        default_index=0,
    )
# home page..............
if selected=="Home":
    st.title(f" {selected} ")
    def weatherSearch(city):
        API_KEY = "9b99c2520d9ddb36ed867de4196e0ede"
        if city_name:
            api_address = "https://api.openweathermap.org/data/2.5/forecast?q=" + city + "&appid=" + API_KEY
            try:
                res = requests.get(api_address)
                a = res.json()
                #   st.write(a)
                
                for i in a['list']:
                    st.subheader(i['dt_txt'])
                    v= i['dt_txt']
                    date_obj = pandas.to_datetime(v).day_name()

                    #   day_name = date_obj.strftime("%A")
                    st.write(date_obj) 
                    col1,col2= st.columns([1,1])
                    with col1:   
                        if i['weather'][0]['description']=="light rain":
                            st.image('rain.jpg', width=100)
                        else:
                            st.image('cloud.jpg', width=100)
                    with col2:    
                        st.write(f":blue[Description] {i['weather'][0]['description']}")
                            
                        st.write(f":blue[Temperature]  {i['main']['temp']-273}° C")
                        st.write(f":Wind Speed  {i['wind']['speed']}km/hr")
                        st.divider()
            except Exception as e:
                st.write("Error is",e)
        
    city_name= st.text_input("Enter City Name")
    weatherSearch(city_name)
    
    
# plant diease page.....................
elif selected=="Plant Diseases" :
    st.title(f" {selected} prediction")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image1=st.image(image, caption="Uploaded Image", use_column_width=True)
        
    classes=['Healthy','Powdery','Rust']
    
    if st.button("Submit"):
        x = process.imageAns(uploaded_image)
        ans=np.argmax(x)
        st.subheader(classes[ans])
elif selected=="Plant Identification" :
    st.title(f" {selected} prediction")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image1=st.image(image, caption="Uploaded Image", use_column_width=True)
        
    # classes=['Healthy','Powdery','Rust']
    classes=['aloevera','banana','bilimbi','cantaloupe','cassava','coconut','corn','cucumber','curcuma','eggplant','galangal','ginger','guava','kale','longbeans','mango','melon','orange','paddy','papaya','peperchili','pineapple','pomelo','shallot','soybeans','spinach','sweetpotatoes','tobacco','waterapple','watermelon']
    if st.button("Submit"):
        x = process.imageAnsId(uploaded_image)
        ans=np.argmax(x)
        st.subheader(classes[ans])
        
        # st.subheader(classes[ans])
        
# crop_yield page...........
elif selected=="Crop Yield":
    st.title(f"{selected} prediction")
    col1,col2,col3=st.columns([1,1,1])
    with col1:
        crop=st.text_input("Crop",placeholder="Enter Crop here..")
        crop_year=st.number_input("Crop_Year")
        season=st.text_input("Season",placeholder="kharif")
        
    with col2:
        state=st.text_input("State",placeholder="Assam")
        area=st.number_input("Area")
        production=st.number_input("Production")
    
    with col3:
        annual_rainfall=st.number_input("Annual_rainfall")
        fertilzer=st.number_input("Fertilizer")
        pesticide=st.number_input("Pesticide")
    

    if st.button("Submit"):
        crop_encoder= joblib.load('crop_enc.pkl')
        crop_encoder_data= crop_encoder.transform([crop])[0]
        season_encoder= joblib.load('season_enc.pkl')
        season_encoder_data= season_encoder.transform([season])[0]
        state_encoder= joblib.load('state_enc.pkl')
        state_encoder_data= state_encoder.transform([state])[0]

        predicted=[crop_encoder_data,crop_year,season_encoder_data,state_encoder_data,area,production,
               annual_rainfall,fertilzer,
               pesticide]
        
        print(predicted)
        model= joblib.load('crop.pkl')
        ans= model.predict([predicted])
        st.subheader(ans[0])
#about page....................... 
elif selected=="About":
    st.title(f"{selected}")
    st.write("""The agrovista focuses on improving agricultural practices using modern technology.
It allows farmers or researchers to upload crop data, soil reports, or plant images through a simple web interface.
Using machine learning and data analysis, the system predicts crop yield, detects plant diseases, and gives farming recommendations.
Our goal is to help farmers make better decisions, increase productivity, and ensure food security

First we discuss about the plant disease .There are so many plant disease in the agriculture field,
so firstly we identify the plant disease and know about there name of  disease.this project helps farmers and researchers detect plant diseases using AI technology.
Users can upload images of plant leaves through a simple web app.
The system analyzes the images and predicts whether the plant is healthy or infected.
Early detection of diseases can help farmers take quick action and save their crops.
The goal is to make farming smarter, healthier, and more productive

Secondly we discuss about the project predicts the yield of crops based on input data such as soil type, weather conditions, and farming practices.
Users can upload crop data through an easy-to-use web app built with Streamlit.
The system uses machine learning models to analyze the data and estimate how much crop will be produced.
Accurate yield predictions help farmers plan better, improve productivity, and reduce risks in agriculture.

""")
    





    



