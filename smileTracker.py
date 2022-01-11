# source https://youtu.be/XIrOM9oP3pA?t=15169
import cv2

classifier_face = cv2.CascadeClassifier("trained_models/haarcascade_frontalface_default.xml")
classifier_smile = cv2.CascadeClassifier("trained_models/haarcascade_smile.xml")


#read webcam
video = cv2.VideoCapture(0)

while True:
    is_successful_read,frame  = video.read()
    if not is_successful_read:
        break

    img_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates = classifier_face.detectMultiScale(img_grayscale)
    
    # scaleFactor = how much the image is reduced or blurred to filter out unwanted image portions
    # minNeighbors = how many squares need to be detected at the same spot to be considered as a detected entity
    # Doc link https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    # smile_coordinates = classifier_simle.detectMultiScale(img_grayscale,scaleFactor=1.7,minNeighbors=20)
    
    # Run face detectopm
    for (x,y,w,h)  in face_coordinates:        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)

        # get the the portion of the face from the original frame
        # this array 'sclicing' can only be done because the variable "frame" is an object of openCV, AND openCV is built upon Numpy
        # which permits the code below. This WILL NOT work on normal python
        the_face= frame[y:y+h,x:x+w] 
        
        # convert the face portion to grays
        img_smile_grayscale = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        # #detect a smile WITHIN A FACE
        smile_coordinates = classifier_smile.detectMultiScale(img_smile_grayscale,scaleFactor=1.7,minNeighbors=20)

        # for (xS,yS,wS,hS)  in smile_coordinates:

        #     # we print the rectangle inside "the_face" which is the portion of the extraced smile image.
        #     cv2.rectangle(the_face,(xS,yS),(xS+wS,yS+hS),(155,170,0),2)

        #draw text on the face square if a smile is detected
        if(len(smile_coordinates) > 0):
            # h=height of the face frame is added to Y so the text is located under the square
            cv2.putText(frame,'Smiling',(x,y+h+35),fontScale=2,
                fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))


    cv2.imshow("Smile detection",frame)
    key = cv2.waitKey(1)

    #If Q key is pressed then quit
    if key == 81 or key == 113:
        break

#release any resources for the video/cam
video.release()
cv2.destroyAllWindows()
