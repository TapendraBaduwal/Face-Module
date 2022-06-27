import face_recognition
import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import json
import streamlit as st
from face_detect import*
from encoding import*

import pyttsx3  #for voice
#initialize Text-to-speech engine
engine = pyttsx3.init()

#This code reduce the time of encoding by passing face to each processor.
def get_encodings(image, boxes, max_worker=3):
    encodesCurFrame = []
    with ProcessPoolExecutor(max_workers=max_worker) as executor:
    # with ThreadPoolExecutor(max_workers=max_worker) as executor:
        results = [
            executor.submit(face_recognition.face_encodings, image, [boxes[i]])
            for i in range(len(boxes))
        ]
    for i in range(len(results)):
        data = results[i].result()
        encodesCurFrame.append(data[0])
    return encodesCurFrame


def recog():
    st.title("Recognize Face")
    cam = cv2.VideoCapture(0) #6 for realsense RGB
    #cam = cv2.VideoCapture('/home/Tapendra/Downloads/rgb_tapendra.avi')

    FRAME_WINDOW = st.image([])

    with open("face_encode.json", "r") as read_file:
        data= json.load(read_file)

    known_encodings = data['encodings']
    #known_encodings = np.array(data['encodings'])
    known_names = data['names']

    
    while True:
        ret, frame = cam.read()
        # convert the input frame from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        #find face locations
        boxes = face_recognition.face_locations(rgb,model='hog')
        #boxes = detect(rgb)
        #print(len(boxes))
        # the facial encodings for faces in the frame
        #t1= time.monotonic()
        #encodings = face_recognition.face_encodings(rgb,boxes)    
        encodings = get_encodings(rgb, boxes)
        #t2 = time.monotonic()
        #print("encoding",(t2-t1)*1000)
        # Loop through found faces in this frame
        for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.6)#How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.

            #Use the known face with smallest distance to the new face
            face_distances = face_recognition.face_distance(known_encodings,face_encoding)

            matchIndex = np.argmin(face_distances)
            if face_distances[matchIndex]< 0.45:
                name = known_names[matchIndex].upper()
                # print(name)
                engine.say(f"Hello how are you {name}")
                # play the speech
                engine.runAndWait()
            else: name = 'Unknown'


            cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left+6, top-12), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

            # face_match_percentage = (1 - face_distances) * 100
            # for i, face_distance in enumerate(face_distances):
            #     print("The test image has a distance of {:.2} from known image {} ".format(face_distance, i))
            #     print("- comparing with a tolerance of 0.6 {}".format(face_distance < 0.6))
            #     print("Face Match Percentage = ", np.round(face_match_percentage, 4))
                
        
        FRAME_WINDOW.image(frame, channels = "BGR") 
        if cv2.waitKey(1) == 13:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__': 
    recog() 