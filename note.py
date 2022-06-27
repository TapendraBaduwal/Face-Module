
#https://learnopencv.com/histogram-of-oriented-gradients/

# Dlib Histogram of Oriented Gradients (HOG) algorithm based

1.Preprocessing of image involves normalising the image but it is entirely optional.

2.compute the image gradient in both the x and y direction
image ko center lagera teyo center ko x ra y direction ko pixal value sanga calculation garne

Gradient magnitude:(horizontal and vertical gradients)
Gx = pixal value above center - pixal value below center pixal
similarly,
Gy = pixal value  left from center - pixal value  right from center 

Finally,
Magnitude of the gradient = Sqrt root of( Gx^2 +Gy^2)
i.e Gradient magnitude matrix form.

And,
Direction of the gradient = arctan(Gy/Gx) i.e inverse of tan function
i.e Gradient direction matrix form.

3.After obtaining the gradient of each pixel, 
the gradient matrices (magnitude and angle matrix) are divided into 8x8 cells to form a block. 
For each block, a 9-point histogram is calculated(features ramro auxa small block lida).

4.Now we can take Gradient direction matrix block and Gradient magnitude matrix block and 
fill magnitude value of matrix in the crossponding position  to its direction angle ,
Final feature vector  is created and   
it forms one histogram from all small histograms which is unique for each face.

5. dlib.get_frontal_face_detector le chai face location ma bbox creat garxa.

6.Boundary boxes baunda image frame matrix  ma object or face vako thau ma different pixel
hunxan so teyo thau ma cluster banxa ra cluster ko centroide ko ori pari Boundary boxes
creat garinxa.

source venv/bin/activate
streamlit run streamlit.py