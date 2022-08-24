import cv2 as cv
import datetime
import numpy as np
from matplotlib import pyplot as plt

motion_delay = 10 # number of seconds after last movement
min_contour_size = 20000

moving = False
last_motion = 0
frame_number = 0
cap = cv.VideoCapture("movement.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print("Video framerate: " + str(fps))
while cap.isOpened():
    diff = cv.absdiff(frame1, frame2)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(
        dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    big_conture = False
    for contour in contours:
        if cv.contourArea(contour) > min_contour_size:
            big_conture = True
            continue

    if big_conture:
        last_motion = frame_number
        if not moving:
            print(frame_number)
            print(str(datetime.timedelta(seconds=frame_number / fps)))
        moving = True

    else:
        if (frame_number - last_motion) > motion_delay*fps:
            moving = False
    # cv.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    frame1 = frame2
    ret, frame2 = cap.read()
    frame_number += 1

cap.release()
cv.destroyAllWindows()
