# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
import io
import cv2
import time
import camera
import datetime
import requests
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from facercogAI import IMGclassifier

# ImageNet 라벨을 가져오는 함수
data_dir = '분류 폴더가 저장된 디렉토리를 입력하세요.'

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.imageAI = IMGclassifier(data_dir)
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []
        self.recognition_log = {"name": [], "time": []}

        # Load sample pictures and learn how to recognize it.
        dirname = '/Users/eunchae/Workspace/python_code/camera_recognition-main/face_recognition/knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if (ext == '.jpg') or (ext == '.jpeg') or (ext == '.png'):
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                #face_encoding = face_recognition.face_encodings(img)[0]
                
                face_encodings = face_recognition.face_encodings(img)
                if not face_encodings:
                    print("No face found in the image.")
                else:
                    face_encoding = face_encodings[0]
    
                self.known_face_encodings.append(face_encoding)
                
        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        
    def log_recognition(self, name):
        current_time = datetime.datetime.now()
        log_entry = {"name": [name], "time": [current_time]}
        #self.recognition_log.append(log_entry)
        self.recognition_log = log_entry

    def __del__(self):
        del self.camera

    def get_frame(self):
        # Grab a single frame of video
        frame = self.camera.get_frame()
        #self.image_segmentation(frame)
        #self.image_classification(frame)
        imgai_res = self.imageAI.run_prediction(frame, threshold=0.85)
        print(f"AI res: {imgai_res}")
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.65:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if (name != "Unknown") or (name == "Unknown") :
                self.log_recognition(name)
           

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

if __name__ == '__main__':
    face_recog = FaceRecog()
    #print(face_recog.known_face_names)
    previous_face_name = {'name': {}}
    vistor_record = pd.DataFrame({"name": [None], "time": [None]})
    target_hour = 23
    target_minute = 46
    while True:
        frame = face_recog.get_frame()
        face_name = face_recog.recognition_log
        if len(face_name) != 0:
            
            if face_name['name'] != previous_face_name['name'] :
                if previous_face_name['name'] is not bool(previous_face_name.get('name')):
                    vistor_record = pd.concat([vistor_record.reset_index(drop = True),
                                               pd.DataFrame(face_name).reset_index(drop = True)
                                              ], ignore_index=True)
                elif previous_face_name['name'] != 'Unknwon':
                    pass
                elif (len(previous_name['name']) != 0) and (face_name['name'] == 0):
                    vistor_record = pd.concat([vistor_record, pd.DataFrame(face_name)], ignore_index=True)
                    
        previous_face_name = face_name
        # show the frame
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
            #break
        # 현재 시간 확인
        current_time = time.localtime()
    
        # 특정 시간에 도달하면 종료
        if current_time.tm_hour == target_hour and current_time.tm_min == target_minute:
            break
            
    my_file_name = '/home/eunchae/workspace/asort_face_video/face_recognition/my_room_visitor.csv'
    vistor_record.to_csv(my_file_name, index = None)
    print('방문자 리스트 저장 완료')
    
    #visitor_record.to_csv('visitor_record.csv', index = None) #csv파일로 생성
    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
