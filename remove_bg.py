from rembg import remove
import cv2
import os

PATH = os.path.dirname(os.path.realpath(__file__))
PATH = PATH + '/'
INPUT_PATH = PATH + "first_test/images/"
OUTPUT_PATH = PATH + "output_2/"

names = os.listdir(INPUT_PATH)

for name in names:
    IMG_PATH = INPUT_PATH + name 
    rembg_img = cv2.imread(IMG_PATH)
    output = remove(rembg_img)
    cv2.imwrite(OUTPUT_PATH + name, output)

