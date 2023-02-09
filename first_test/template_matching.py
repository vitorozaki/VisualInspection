# Python program to illustrate
# template matching
import cv2
import numpy as np
import os

# Template matching methods
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

method = eval(methods[5])
i = 1
images_names = os.listdir("./images")
for image in images_names:
    # Read the main image
    img_rgb = cv2.imread("./images/" + image)

    # Convert it to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Read the templates
    template_names = os.listdir("./base_img")
    for name in template_names:
        template = cv2.imread("./base_img/" + name, 0)

        # Store width and height of template in w and h
        w, h = template.shape[::-1]

        # Perform match operations.
        res = cv2.matchTemplate(img_gray, template, method)

    # # Specify a threshold
    # threshold = 0.4
    
    # # Store the coordinates of matched area in a numpy array
    # loc = np.where(res >= threshold)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # print(len(loc))
    
    # # Draw a rectangle around the matched region.
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img_rgb ,top_left, bottom_right, 255, 2)

    # Save the final image with the matched area.
    cv2.imwrite(f'Detected_{i}.jpg', img_rgb)
    i += 1

