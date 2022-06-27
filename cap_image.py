
import cv2
import os 
import streamlit as st
from encoding import*

path = '/home/tapendra/Desktop/StreamlitFaceModule/dataset/'

def capture():    
    """Function to capture images
    """
    st.title("Capture Image")
    col1, col2 = st.columns([3, 2])
    col1.header("Live video")
    col2.header("Captured")
    with col1:
        FRAME_WINDOW1 = st.image([])
        name = st.text_input('Enter name and press enter to capture image', key ="9")
    with col2:
        FRAME_WINDOW2 = st.image([])
        save = st.button("Save & change Image")
    
    camera = cv2.VideoCapture(0)
    while True:
        ret ,frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if save:
            rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            while save:
                cv2.imwrite(path + name + '.png',rgb)
                encoding()                   #from face_recog import * 
                col2.image(frame, use_column_width=True)  #Note show only image in col2           

                save = False
      
        FRAME_WINDOW1.image(frame)
        if cv2.waitKey(1) == 13:
            break


if __name__ == '__main__':  
    capture()           