import cv2
import numpy as np
import PIL
import os
from utils.functions import denoise, cv_seg, multi_dil, count, erase_dir
from rembg import remove

import time

path = os.path.dirname(os.path.realpath(__file__))
path = path + '/'

save_img = "extractions/"

ip = "http://192.168.15.2:4747/video"

font = cv2.FONT_HERSHEY_SIMPLEX
# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(ip) 

width = int(cap.get(3))
height = int(cap.get(4))

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

start_time = time.time()
num = 0

while (True):
    current_time = time.time()
    ret, frame = cap.read()
    frame = frame[15:height,:]

    if (round((current_time - start_time) % 2) == 0):
        # Remove background
        rembg_img = remove(frame)

        # # Denoise image
        # denoised = denoise(rembg_img)

        # Convert to grayscale
        gray = cv2.cvtColor(rembg_img, cv2.COLOR_BGR2GRAY)

        # Segment image
        seg_img = cv_seg(gray)

        # # erode imge
        # eroded = multi_ero(seg_img, 2) 

        # Dilate image
        dilated = multi_dil(seg_img, 2)

        # Count and locate objects
        objs =  count(dilated)
        num = len(objs)

        erase_dir(path + save_img)
        # Draw boxes over image
        for i, obj in enumerate(objs):
            x0 = obj[0][0]
            y0 = obj[0][1]
            x1 = obj[1][0]
            y1 = obj[1][1]
            # print(f"x0 = {x0}\nx1 = {x1}\ny0 = {y0}\ny1 = {y1}")
            roi = frame[y1:y0, x0:x1] 
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.imwrite(path+save_img+ f"{i}.jpg", roi)

    # Display number of objects
    cv2.putText(frame, f"Objects: {num}", (10, 50), 
                fontFace=font, 
                fontScale=1, 
                color=(255, 0, 0),
                thickness=1)

    
    # time when we finish processing for this frame
    new_frame_time = time.time()
  
    # Calculating the fps
  
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = float(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
  
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (10, 20), 
    fontFace=font, 
    fontScale=0.5,
    color=(100, 255, 0),
    thickness=2)
    

    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # esc
        break