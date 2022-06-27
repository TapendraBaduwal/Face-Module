from imutils import paths
import face_recognition
import cv2
import os
import json

def encoding():
    imagePaths = list(paths.list_images('dataset'))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        #name = imagePath.split(os.path.sep)[-2] #dataset/Tapendra/Tapendra.png ma -2 index vaneko Tapendra ho folder banayera image save garda.
        name = imagePath.split(os.path.sep)[1]
        #print(name)
        name =name.split(".")
        name = name[0]
        #print(name)
        
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb, model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            #knownEncodings.append(encoding)
            knownEncodings.append(encoding.tolist())
            knownNames.append(name)
            

    #save encodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #data = {"encodings": knownEncodings.tolist(), "names": knownNames}
    #data = {"encodings": knownEncodings.tolist(), "names": knownNames.tolist()}

    ######## Method-1 ###########
    #Serializing json 
    # json_object = json.dumps(data, indent = 4)
    # # #Writing to face_encode.json
    # with open("face_encode.json", "w") as write_file:
    #     write_file.write(json_object)


    ######## Method-2 ###########
    with open("face_encode.json","w") as write_file:  
        json.dump(data, write_file)  

if __name__=="__main__":
    encoding()