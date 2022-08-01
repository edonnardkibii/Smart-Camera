"""

Author: James Kibii
Project: Security & IoT
Description: Send an encrypted web stream from one device to another

"""

import cv2
import paho.mqtt.client as mqtt
import base64
import time
import configparser
import subscription
# import certifi
from cryptography.fernet import Fernet

# import numpy as np
# from Crypto.Cipher import AES
# from Crypto import Random
# import os
# import sys

# MQTT_BROKER = "test.mosquitto.org"
# MQTT_BROKER = "broker.hivemq.com"

class VideoStream(object):
    def __init__(self):
        self.__video = cv2.VideoCapture(0)
        self.__client = None
        # self.__MQTT_BROKER = "test.mosquitto.org"
        self.__MQTT_BROKER = "broker.hivemq.com"
        self.__PORT = 1883
        # self.__PORT = 8883
        self.__MQTT_SEND_VIDEO = subscription.topic["subscribe video"]
        self.__cipher = self.__create_cipher()

    @staticmethod
    def __create_cipher():
        # Read config file
        config = configparser.ConfigParser()
        config.read('secret_key.ini')
        secretKey_str = config['Symmetric Encryption']['Secret Key']
        secretKey = base64.b64decode(secretKey_str.encode('utf-8'))
        cipher = Fernet(secretKey)
        return cipher


    def __encrypt(self, msg, cipher):
        cipher_msg = cipher.encrypt(msg)
        return cipher_msg


    def connect(self):
        if self.__client is None:
            self.__client = mqtt.Client()
            self.__client.connect(self.__MQTT_BROKER, self.__PORT)

    def send(self):
        try:
            print("Streaming Live-Video")
            # secretKey = self.__load_secret_key()

            while True:
                start = time.time()
                #self.__video(3, 640)
                # self.__video(4, 480)
                _, frame = self.__video.read()
                # cv2.imshow('Movie', frame)
                # Convert video to grey color
                # grey_video = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # _, buffer = cv2.imencode('.jpg', grey_video)
                # print(frame)
                _, buffer = cv2.imencode('.jpg', frame)
                video_data = base64.b64encode(buffer)

                encrypted_video_data = self.__encrypt(video_data, self.__cipher)
                # secure_video_data = base64.b64encode(encrypted_video_data)
                self.__client.publish(self.__MQTT_SEND_VIDEO, encrypted_video_data)
                
                
                end = time.time()
                t = end - start
                fps = 1 / t
                # print("FPS: " +str(fps))
        except:
            self.__video.release()
            print("Stopping Video Transmission")
            self.__client.disconnect()

if __name__ == '__main__':
    try:
        video_sender = VideoStream()
        video_sender.connect()
        video_sender.send()
    except KeyboardInterrupt:
        print('Stopping Video Transmission')