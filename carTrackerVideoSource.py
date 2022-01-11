# source https://youtu.be/XIrOM9oP3pA?t=8199
#mp4 source and haar cascade model from https://github.com/misbah4064/car_detector_haarcascades
import cv2

videoFilePath = "images/bike_city.mp4"
#file taken from https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml
classifier_file_cars = "trained_models/cars_detection.xml"
classifier_file_pedestrian = "trained_models/haarcascade_full_body.xml"

classifier_cars = cv2.CascadeClassifier(classifier_file_cars)
classifier_pedestrians = cv2.CascadeClassifier(classifier_file_pedestrian)


#read video and  convert to grayscale
video = cv2.VideoCapture(videoFilePath)

while True:
    is_successful_read,frame  = video.read()
    if not is_successful_read:        
        break
    
    img_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    car_coordinates = classifier_cars.detectMultiScale(img_grayscale)
    pedestrian_coordinates = classifier_pedestrians.detectMultiScale(img_grayscale)

    for coordinates in car_coordinates:
        (x,y,w,h) = coordinates
        #draw rectangle on detected item
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.rectangle(frame,(x+1,y+2),(x+w,y+h),(255,0,0),2)

    for coordinates in pedestrian_coordinates:
        (x,y,w,h) = coordinates
        #draw rectangle on detected item
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Car detection",frame)
    key = cv2.waitKey(1)

    #If Q key is pressed then quit
    if key == 81 or key == 113:
        break

#release any resources for the video/cam
video.release()
