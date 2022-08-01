"""

Author: James Kibii
Project: Security & IoT
Description: Create an MQTT Publisher & try send & receive vidoes

"""

import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import configparser
import subscription
import argparse
import imutils
import time
import os
import ssl
from cryptography.fernet import Fernet

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


class VideoStream(object):
    def __init__(self,):
        self.__frame = np.zeros((240, 320, 3), np.uint8)
        self.__MQTT_RECEIVE = subscription.topic["subscribe video"]
        # self.__MQTT_BROKER = "test.mosquitto.org"
        self.__MQTT_BROKER = "broker.hivemq.com"
        self.__PORT = 1883
        # self.__PORT = 8883
        self.__args = None
        self.__cipher = self.__create_cipher()
        # self.__face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    @staticmethod
    def __create_cipher():
        # Read config file
        config = configparser.ConfigParser()
        config.read('secret_key.ini')
        secretKey_str = config['Symmetric Encryption']['Secret Key']
        secretKey = base64.b64decode(secretKey_str.encode('utf-8'))
        cipher = Fernet(secretKey)
        return cipher

    def __decrypt(self, encrypted_msg):
        decrypted_msg = self.__cipher.decrypt(encrypted_msg)
        return decrypted_msg

    def __connect(self) -> mqtt:
        def on_connect(client, userdata, flags, rc):
            print("Connected with result code " + str(rc))
            client.subscribe(self.__MQTT_RECEIVE)

        client = mqtt.Client()
        client.on_connect = on_connect
        client.connect(self.__MQTT_BROKER, self.__PORT)
        return client

    def __read_message(self, client: mqtt):
        def on_message(client, userdata, msg):
            token = self.__decrypt(msg.payload)
            img = base64.b64decode(token)
            npimg = np.frombuffer(img, dtype=np.uint8)
            self.__frame = cv2.imdecode(npimg, 1)

        client.on_message = on_message

    def __detect_mask(self, frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []  # faces
        locs = []   # locations
        preds = []  # predictions

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.__args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                if face.any():
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    def __define_args(self):
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--face detector config", type=str, default="face_detector",
                        help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                        help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
        self.__args = vars(ap.parse_args())

    def __load_model(self):
        # load the serialized face detector model
        print("Loading face detector model...")
        prototxtPath = os.path.sep.join([self.__args["face detector config"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([self.__args["face detector config"],
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        # gray = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)
        # faceNet = self.__face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(10,10),
        #                                            flags=cv2.CASCADE_SCALE_IMAGE)
        # load the face mask detector model from disk
        print("Loading face mask detector model...")
        maskNet = load_model(self.__args["model"])

        return faceNet, maskNet

    '''
    def __detect_face(self):
        gray = cv.cvtColor(self.__frame, cv.COLOR_BGR2GRAY)

        faces = self.__face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9, minSize=(10,10),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
        for (x,y, w, h) in faces:
            cv.rectangle(self.__frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = self.__frame[y:y+h, x:x+w]

            eyes = self.__eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 127, 255), 2)
    '''
    def __stream(self):
        self.__define_args()
        faceNet, maskNet = self.__load_model()
        while True:
            frame = imutils.resize(self.__frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = self.__detect_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # self.__detect_face()
            cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # client.loop_stop()

    def run(self):
        client = self.__connect()
        self.__read_message(client)
        client.loop_start()
        self.__stream()
        client.loop_stop()

if __name__ == '__main__':
    video_receiver = VideoStream()

    try:
        video_receiver.run()
    except KeyboardInterrupt:
        print('Stopping')

