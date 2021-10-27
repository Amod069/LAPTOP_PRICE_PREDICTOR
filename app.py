import streamlit as st
import pickle
import numpy as np
# import model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
st.title("LAPTOP PRICE PREDICTOR")
#company
company= st.selectbox('LAPTOP BRANDS',df['Company'].unique())
#type name
type= st.selectbox('TYPE OF LAPTOP',df['TypeName'].unique())
#ram
ram= st.selectbox('RAM(in GB)',df['Ram'].unique())
#weight
weight=st.number_input('Weight in Kgs')
#touchscreen or not
touchscreen=st.selectbox('TouchScreen',['NO','YES'])
#ips or not
ips=st.selectbox('IPS Display',['NO','YES'])
#screensize
screensize=st.number_input('ScreenSize in Inches')
#resolution
resolution=st.selectbox('Screen Resolution',['1366x768','1600x900','1920x1080',
                                             '2304x1440','2560x1440','2560x1600',
                                             '2880x1800','3000x2000',
                                             '3200x1800','3840x2160'])
#cpu
cpubrand= st.selectbox('Processor Brand',df['Cpu brand'].unique())
#hard drive
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
#SSD
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
#graphic
gpubrand= st.selectbox('Graphic Card Brand',df['Gpu Name'].unique())
#OS
os= st.selectbox('Operating System',df['os'].unique())

if st.button('Predict Price'):
    if touchscreen=='YES':
        touchscreen=1
    else:
        touchscreen = 0
    if ips=='YES':
        ips=1
    else:
        ips = 0

    ppi= None
    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=(((X_res**2)+(Y_res**2))**0.5)/screensize
     # Query Point
    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpubrand,hdd,ssd,gpubrand,os])
    query=query.reshape(1,12)
    st.title("PREDICTED PRICE BY MODEL IS " +str(round(int(np.exp(pipe.predict(query))))))