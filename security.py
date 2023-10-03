#IMAGE STUFF

# import cv2

# img = cv2.imread('assets/setup.png', -1)
# img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
# cv2.imshow("Image", img)

# # Save new image as file name 
# # cv2.imwrite("smaller_setup.png", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows



# VIDEO STUFF

import numpy as np
import cv2
import datetime
import time


# Initialize variables for timing and detection.
detection = False
timeSinceDetection = None
timerStarted = False
recordAfterDetectionSeconds = 5



# Correct the file names and paths
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
bodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_fullbody.xml")
upperBodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

# Videocapture
cap = cv2.VideoCapture(0)

frameSize = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    # Only worried about getting the one frame.
    _, frame = cap.read()

    # Cascades use grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Finds parts of the frame that include bodies or faces - scaleFactor is the accuracy and minNeighbors is how sure the detection needs to be
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    bodies = bodyCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3)
    upperBodies = upperBodyCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3)

    # If there is a person in frame
    if len(upperBodies) > 0 or len(bodies) > 0 or len(faces):
        # If we are already detecting and a body comes in, reset the timer
        if detection:
            timerStarted = False
        # If not, start the video.
        else:
            detection = True
            currentTime = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{currentTime}.mp4", fourcc, 20, frameSize)
            print('started recording')
    

    # If detecting but no person in frame
    elif detection:
        # If timer is started, count down from 5 seconds before ending
        if timerStarted:
            # if current - started > 5 then stop the timer
            if time.time() - timeSinceDetection >= recordAfterDetectionSeconds:
                detection = False
                timerStarted = False
                out.release()
                print('stopped recording.')
        # If timer not started, start it.
        else:
            timerStarted = True
            # start timer
            timeSinceDetection = time.time()

    # if it's detected, write the current frame to the 
    if detection:
        out.write(frame)

    # show frame for testing - not necessary
    cv2.imshow('frame', frame)

    # If q pressed, stop the input stream.
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()