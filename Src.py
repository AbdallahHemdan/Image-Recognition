#Face_Eye_Detection
#Author:Abdallah_Hemdan 

#  ___  __
# |\  \|\  \
# \ \  \_\  \
#  \ \   ___ \EMDAN
#   \ \  \\ \ \
#    \ \__\\ \_\
#     \|__| \|__|

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detects objects of different sizes in the input image.
    #The detected objects are returned as a list of rectangles
    #detectMultiScale(img,ScaleFactor,minNeighbours)
    #scaleFactor : Parameter specifying how much the image size is reduced at each image scale
    #minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew-5,ey+eh-5),(0,0,255),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
