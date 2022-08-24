import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def motionDetection():
    frameNumber = 0
    cap = cv.VideoCapture("movement.mp4")
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv.absdiff(frame1, frame2)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour) > 900:
                print(frameNumber)
                continue


        # cv.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        frame1 = frame2
        ret, frame2 = cap.read()
        frameNumber += 1

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    motionDetection()
