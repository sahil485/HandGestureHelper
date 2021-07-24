import cv2
import tensorflow as tf
import numpy as np
import time
from pynput.mouse import Controller as mouseController
from pynput.keyboard import Key, Controller
mouse = mouseController()
key = Controller()
network = tf.keras.models.load_model("gestures_modelv6")

def classification(image):
    global network
    prediction = network.predict(image)
    if prediction[0][1]+prediction[0][2]+ prediction[0][3]<0.95 and prediction[0][0]>.20:
        return 0
    elif max(prediction[0]) == prediction[0][1] and prediction[0][1]>prediction[0][2]+0.25:
        return 3
    elif max(prediction[0]) == prediction[0][3]:
        return 2
    else:
        return 1

def none(predict): True if predict[0][1]+predict[0][2]<0.95 else False


def thumbUp(predict): True if max(predict[0]) == predict[0][1] else False


def thumbDown(predict): True if max(predict[0]) == predict[0][2] else False

video_feed = cv2.VideoCapture(0)

cTime = 0

while True:
    success, frame = video_feed.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (thresh, monoFrame) = cv2.threshold(frame, 70, 255, cv2.THRESH_BINARY_INV)


    cv2.imshow("win", monoFrame)
    cv2.waitKey(25)

    monoFrame = cv2.resize(monoFrame, (40, 30))
    monoFrame = np.reshape(monoFrame, [1, 40, 30, 1])

    decision = classification(monoFrame)

    if decision == 1:
        mouse.scroll(0,0.5)
        print("thumbUp")
    elif decision == 2:
        mouse.scroll(0,-0.5)
        print("thumbDown")
    elif decision == 3:
        key.press(Key.ctrl)
        key.press(Key.tab)
        key.release(Key.ctrl)
        key.release(Key.tab)
        print("forward")
    else:
        print("none")