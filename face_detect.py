import cv2
import mediapipe as mp
import streamlit as st

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect():
    """Function to detect faces using mediapipe
    """
    st.title("Detect Face")
    camera = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        while True:
            ret, frame = camera.read()          
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            image = frame.copy()
            results = face_detection.process(image)
            if results.detections:
                for landmark in results.detections:
                    mp_drawing.draw_detection(image, landmark)
                FRAME_WINDOW.image(image)   
            else:
                 FRAME_WINDOW.image(image)
                 
        camera.release()
        cv2.destroyAllWindows()
     

if __name__ == '__main__': 
    detect() 


#It is based on BlazeFace
#the major six-coordinate for every face it will detect in the image/frame. Those six coordinates are as follows:
#Right Eye,Left Eye,Nose Tip,Mouth Center,Right Ear Tragion,Left Ear Tragion                    