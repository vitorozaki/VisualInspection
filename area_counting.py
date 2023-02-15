import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.functions import prep_img
from skimage.measure import label, regionprops
import time

ip = "http://192.168.15.5:4747/video"

font = cv2.FONT_HERSHEY_SIMPLEX
# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(ip) 

width = int(cap.get(3))  # 640
height = int(cap.get(4))  # 480

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

start_time = time.time()


""" Save video 
Save video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 9, (width, height - 15))
"""

# Creating rectangle sections
sections = []
for x in range(1, 4):
    for y in range(1, 4):
        x0 = int(width - (width / 3 * x)) + 6
        x1 = int(x0 + 200)
        y0 = int(height - (height / 3 * y)) + 4
        y1 = int(y0 + 150)
        sections.append((y0, y1, x0, x1))

while (True):
    current_time = time.time()
    ret, frame = cap.read()
    final_frame = prep_img(frame)
    for section in sections:
        cv2.rectangle(frame, (section[2], section[0]), (section[3], section[1]), (0, 0, 0), 1)
        roi = final_frame[section[0]:section[1], section[2]:section[3]]

        # Count and locate objects
        label_im = label(roi)
        count = 0
        actual_img = []  # stores the objects detected in the image
        for i in regionprops(label_im):
            y1, x0, y0, x1 = i.bbox
            cv2.rectangle(frame, (x0 + section[2], y0 + section[0]), (x1 + section[2], y1 + section[0]), (0, 0, 255), 1)
            count += 1

        # Display number of objects
        cv2.putText(frame, f"Objects: {count}", (section[2], section[0]+10), 
                    fontFace=font, 
                    fontScale=0.3, 
                    color=(255, 0, 0),
                    thickness=1)

    
    # fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = float(fps)
    fps = str(fps)
  
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (10, 20), 
    fontFace=font, 
    fontScale=0.5,
    color=(100, 255, 0),
    thickness=2)
    
    # out.write(frame)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # esc
        break