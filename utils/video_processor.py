import cv2
import numpy as np

def extract_frames(video_path, interval=30):

    cap = cv2.VideoCapture(video_path)

    frames=[]

    count=0

    while True:

        success,frame=cap.read()

        if not success:
            break

        if count%interval==0:
            frames.append(frame)

        count+=1

    cap.release()

    return frames