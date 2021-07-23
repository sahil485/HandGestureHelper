import cv2
import tensorflow as tf
import numpy as np
from pynput.mouse import Controller

mouse = Controller()
network = tf.keras.models.load_model("gestures_modelv4")

def classification(image):
    global network
    prediction = network.predict(image)
    if prediction[0][1]+prediction[0][2]<0.95:
        return 0
    elif max(prediction[0]) == prediction[0][1]:
        return 1
    else:
        return 2

def none(predict): True if predict[0][1]+predict[0][2]<0.95 else False


def thumbUp(predict): True if max(predict[0]) == predict[0][1] else False


def thumbDown(predict): True if max(predict[0]) == predict[0][2] else False

video_feed = cv2.VideoCapture(0)


while True:
    success, frame = video_feed.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (thresh, monoFrame) = cv2.threshold(frame, 70, 255, cv2.THRESH_BINARY_INV)


    cv2.imshow("win", monoFrame)
    cv2.waitKey(50)

    monoFrame = cv2.resize(monoFrame, (40, 30))
    monoFrame = np.reshape(monoFrame, [1, 40, 30, 1])

    decision = classification(monoFrame)

    if decision == 1:
        mouse.scroll(0,0.5)
        cv2.putText(monoFrame, "thumbUp", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 10)
        print("thumbUp")
    elif decision == 2:
        mouse.scroll(0,-0.5)
        print("thumbDown")
    else:
        print("none")