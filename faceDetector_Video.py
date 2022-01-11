#Notes: RGB stands for Red, Green, Blue; that's how colors are often defined on Software development
# OpenCV Handles the color names as BGR, it's the same thing, just in dofferent order

import cv2


#load classifier data, holds a trained model of what a face is 
trained_face_data = cv2.CascadeClassifier('trained_models/haarcascade_frontalface_default.xml')

#Choose video source to detect faces in 
webcam = cv2.VideoCapture(0)

#because we are running on webcam, we run the code forever
while True:
    #read frames from webcam
    #is_successful_read is bool
    #frame = contains frame data.
    is_successful_read,frame  = webcam.read()
    img_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #Face detection code
    face_coordindates = trained_face_data.detectMultiScale(img_grayscale)
    for coordinates in face_coordindates:
        # #assign to a tuple (x,y,w,h) whatever coordinates were extracted from the multiscale detection
        (x,y,w,h) = coordinates
        #draw rectangle on detected item
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('single face example',frame)
    
    #We add a delay of 1 second to wait key, this means that it will capture 1 frame each 1 (one) millisecond
    key = cv2.waitKey(1)

    #If Q key is pressed then quit
    if key == 81 or key == 113:
        break
#reslease webcan resource
webcam.release()

# key = cv2.waitKey(1)

