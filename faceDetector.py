#Notes: RGB stands for Red, Green, Blue; that's how colors are often defined on Software development
# OpenCV Handles the color names as BGR, it's the same thing, just in dofferent order

# CLassifier link  https://github.com/opencv/opencv/tree/4.x/data/haarcascades
# OpenCV documentation https://docs.opencv.org/4.x/index.html

import cv2

#load classifier data, holds a trained model of what a face is 
trained_face_data = cv2.CascadeClassifier('trained_models/haarcascade_frontalface_default.xml')

#Choose image to detect faces in 
img = cv2.imread('images/multiple_faces.jpg')



#convert image to grayscale, this is needed to process the image.
#Color images need a different algorithm if you want to detect faces or whatever
img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#Face detection code
face_coordindates = trained_face_data.detectMultiScale(img_grayscale)
#These are the coordinates found for each detected face
print(face_coordindates)

#Drawing rectanges into each detected face
for coordinates in face_coordindates:
    # #assign to a tuple (x,y,w,h) whatever coordinates were extracted from the multiscale detection
    (x,y,w,h) = coordinates
    #draw rectangle on detected item
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('single face example',img)
cv2.waitKey()

print("Finished")
