# source https://youtu.be/XIrOM9oP3pA?t=8199
#mp4 source and haar cascade model from https://github.com/misbah4064/car_detector_haarcascades
import cv2


imgFilePath = "images/cars.jpg"

#file taken from https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml
classifier_file = "trained_models/cars_detection.xml"

cars_classifier = cv2.CascadeClassifier(classifier_file)

#read image and  convert to grayscale
img = cv2.imread(imgFilePath)
img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect cars and draw on image
car_coordinates = cars_classifier.detectMultiScale(img_grayscale)
for coordinates in car_coordinates:
    # #assign to a tuple (x,y,w,h) whatever coordinates were extracted from the multiscale detection
    (x,y,w,h) = coordinates
    #draw rectangle on detected item
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("Car classfier",img)

cv2.waitKey()
print("It works!")