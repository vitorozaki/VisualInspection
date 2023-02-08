import cv2
import numpy as np
import PIL
import os
from utils.functions import denoise, cv_seg, multi_dil, erase_dir, bin_img, prep_img
from rembg import remove
from skimage.measure import label, regionprops


import time

path = os.path.dirname(os.path.realpath(__file__))
path = path + '/'

save_img = "base_img/"

ip = "http://192.168.15.2:4747/video"

font = cv2.FONT_HERSHEY_SIMPLEX
# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(ip) 

width = int(cap.get(3))
height = int(cap.get(4))

while (True):

    erase_dir(path + save_img)

    current_time = time.time()
    ret, frame = cap.read()
    frame = frame[15:height,:]

    final_frame = prep_img(frame)

    # Count and locate objects
    label_im = label(final_frame)
    count = 0
    boxes = []
    for i in regionprops(label_im):
        minr, minc, maxr, maxc = i.bbox

        x0 = minc
        y0 = maxr
        x1 = maxc
        y1 = minr

        box = [y1, y0, x1, x0]
        boxes.append(box)
        # roi = frame[y1:y0, x0:x1] 
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # cv2.imwrite(path+save_img+ f"{i}.jpg", roi)
        count += 1

    # Display number of objects
    cv2.putText(frame, f"Objects: {count}", (10, 50), 
                fontFace=font, 
                fontScale=1, 
                color=(255, 0, 0),
                thickness=1)


    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff

    if k == 32:
        for i, box in enumerate(boxes):
            roi = frame[box[0]+1:box[1], box[3]+1:box[2]]
            cv2.imwrite(path+save_img+ f"{i}.jpg", roi)
        break
    
    if k == 27:  # esc
        break