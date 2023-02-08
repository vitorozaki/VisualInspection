import cv2
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
from utils.functions import prep_img, histDist, shape_distance, shape_similarity
from rembg import remove
from skimage.measure import label, regionprops
import time

path = os.path.dirname(os.path.realpath(__file__))
path = path + '/'
base_imgs_path = "base_img/"

# open base images
base_images = []
base_images_names = os.listdir(path + base_imgs_path)
base_images_names.sort()

# Base images labels
labels = ["case", "bottle cap", "red pen", "blue pen", "green coin"]

for name in base_images_names:
    base_images.append(cv2.imread(path+base_imgs_path+name))

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


""" Save video 
Save video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 9, (width, height - 15))
"""

while (True):
    current_time = time.time()
    ret, frame = cap.read()
    frame = frame[15:height,:]

    final_frame = prep_img(frame)

    # Count and locate objects
    label_im = label(final_frame)
    count = 0
    actual_img = []  # stores the objects detected in the image
    for i in regionprops(label_im):
        minr, minc, maxr, maxc = i.bbox

        x0 = minc
        y0 = maxr
        x1 = maxc
        y1 = minr

        roi = frame[y1-1:y0, x0-1:x1] 
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
        count += 1

        aux = []  # stores the histograms distances
        for base_img in base_images:
            # Compare the shape from test object with every base object
            # and if the difference is below 7%, calculate distance 
            # between histograms
            if (shape_distance(base_img, roi) >= 0.8):

                # compare histograms and store into aux list
                res = histDist(base_img, roi) 
                aux.append(res)

            # stores 1 for images with different shape
            else:
                aux.append(1)
                
        # From histogram distances, the minimum value's index is used to locate the name of
        # the object from the list of names
        minimum = min(aux)
        index = aux.index(minimum)
        match = labels[index]
        if minimum != 1 and minimum <= 0.55:
            cv2.putText(frame, match, (x0, y0), 
                fontFace=font, 
                fontScale=1, 
                color=(255, 0, 0),
                thickness=1)
            actual_img.append(match)

        # missing = [element for element in labels if element not in actual_img]
    # print(f"found objects: {actual_img}")
    print(f"missing objects: {list(element for element in labels if element not in actual_img)}")
    cv2.putText(frame, f"{list(element for element in labels if element not in actual_img)}", (0, height-20), 
                    fontFace=font, 
                    fontScale=1, 
                    color=(255, 0, 0),
                    thickness=1)


    # Display number of objects
    cv2.putText(frame, f"Objects: {count}", (10, 50), 
                fontFace=font, 
                fontScale=1, 
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