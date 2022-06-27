import streamlit as st
from cap_image import*
from face_detect import*
from face_recog import*

st.title("Streamlit Tutorial")
html_temp = """
<body style="background-color:red;">
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit  Face Detection and Recongnition </h2>
</div>
</body>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.title("Face Module")

add_selectbox = st.sidebar.selectbox("What would you like to perform?",
    ("Face Capture","Face Detection", "Face Recognition"))

# Face Capture
if add_selectbox == "Face Capture":      
    run = st.checkbox('run',key= "1")
    while run:  
        capture()   

# Face Detection
elif add_selectbox == "Face Detection":  
    run = st.checkbox('Run',key= "2")
    while run:
        detect()               


# Face Recognition
elif add_selectbox == "Face Recognition":
    run = st.checkbox('Run',key= "3")
    while run:
        recog()

    

