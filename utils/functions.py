import cv2
import numpy as np
import PIL
import os
import shutil

# denoise
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float


# Dilation and erosion
from skimage.morphology import dilation, erosion
from skimage.morphology import rectangle

# Blob count
from skimage.measure import label, regionprops

def denoise(img):
    """ Converts the numpy array to float to apply non-local
        means denoising, then converts back to 8 byte int.
        Must be applied over >2-dim image, graysacale images
        won't work.
            
        ----------------------
        Parameters

            img : numpy.ndarray
                input RGB image
        ----------------------
        Returns
        
            numpy.ndarray
                Denoised image in 8 byte int format
    """
    float_img = img_as_float(img)
    sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=True, 
                                patch_size=5, patch_distance=3, channel_axis=-1)
    denoise_img_as_8byte = img_as_ubyte(denoise_img)
    # gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
    return denoise_img_as_8byte


def cv_seg(img):
    """ Segment image with binary and triangle thresholding

        ----------------------
        Parameters

            img  :  numpy.ndarray
                input grayscale image
        ----------------------
        Returns

            numpy.ndarray
                Segmented image with pixel values equal to 0 (black)
                or 255 (white) only.
    """
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    return thresh


def multi_dil(img, num):
    """ Apply the dilation operation, shrinking dark region and enlarging 
        bright areas. For better results, should be applied over segmented
        images

        ----------------------
        Parameters

            img  :  numpy.ndarray
                input segmented image
            num  :  int
                number of dilations to apply over the image
    
        -----------------------
        Returns

            numpy.ndarray
    
    """
    for i in range(num):
        img = dilation(img)
    return img

def multi_ero(im,num):
    for i in range(num):
        im = erosion(im, rectangle(2, 7))
    return im

def count(img):
    """ Count and locate the objects on the image 

        ----------------------
        Parameters

            img  :  numpy.ndarray
                input image
        ----------------------

        Returns

            List
                List containing the boxes' coordinates of each
                Blob (object) detected. The list's length is the
                number of objects
    """
    label_im = label(img)
    boxes = []
    for i in regionprops(label_im):
        minr, minc, maxr, maxc = i.bbox
        # # plt rectangle
        # rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                       fill=False, edgecolor='red', linewidth=1)
        # boxes.append(rect)

        # OpenCV box
        box = ((minc, maxr), (maxc, minr))
        boxes.append(box)

    return boxes


def erase_dir(folder):
    """ Deletes all files inside given folder
    
    ----------------------
    Parameters

        folder: string

            path to the folder to be emptied 

    ----------------------
    Returns

        None

    """

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))