import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

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
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh


def bin_img(img):
    """ Converts a segmented image with range [0, 255] to [0, 1]

    ----------------------
    Parameters

        img: numpy.ndarray
            input image
    
    ----------------------
    Returns

        numpy.ndarray
    
    """
    aux = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
    aux[(img[:] != 255)] = 1

    return aux


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
    """ Apply the erosion operation, shrinking bright region and enlarging 
        dark areas. For better results, should be applied over segmented
        images

        ----------------------
        Parameters

            img  :  numpy.ndarray
                input segmented image
            num  :  int
                number of erosions to be applied over the image
    
        -----------------------
        Returns

            numpy.ndarray
    
    """
    for i in range(num):
        im = erosion(im, rectangle(2, 7))
    return im


def prep_img(img):
    """ Apply all filters to equalize image and prepare for blob detections

        -----------------------
        Parameters

            img: np.ndarray
                input img

        -----------------------
        Returns

            np.ndarray
                   
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise image
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 11)

    # Segment image
    segmented = cv_seg(denoised)

    # Binarize image
    binarized = bin_img(segmented)

    # Dilate image
    dilated = multi_dil(binarized, 6)

    return dilated




# def count(img):
#     """ Count and locate the objects on the image 

#         ----------------------
#         Parameters

#             img  :  numpy.ndarray
#                 input image
#         ----------------------

#         Returns

#             List
#                 List containing the boxes' coordinates of each
#                 Blob (object) detected. The list's length is the
#                 number of objects
#     """
#     label_im = label(img)
#     count = 0
#     for i in regionprops(label_im):
#         minr, minc, maxr, maxc = i.bbox
#         # # plt rectangle
#         # rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
#         #                       fill=False, edgecolor='red', linewidth=1)
#         # boxes.append(rect)

#         # OpenCV box
#         box = ((minc, maxr), (maxc, minr))
#         x0 = box[0][0]
#         y0 = box[0][1]
#         x1 = box[1][0]
#         y1 = box[1][1]
#         # print(f"x0 = {x0}\nx1 = {x1}\ny0 = {y0}\ny1 = {y1}")
#         # roi = frame[y1:y0, x0:x1] 
#         frame = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
#         # cv2.imwrite(path+save_img+ f"{i}.jpg", roi)
#         count += 1
#     return count, frame


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


def crop(img_path, output_path):
    """ Saves cropped objects into a folder

    ----------------------
    Parameters

        img_path: String
            path to image to be used as base
    
        output_path: String
            Folder to which the cropped objects will be saved

    ----------------------
    Returns

        None

    """

    img = cv2.imread(img_path)
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    rembg_img = remove(resized)
    denoised = denoise(rembg_img)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    segmented = cv_seg(gray)
    dilated = multi_dil(segmented, 3)
    label_im = label(segmented)
    for num, i in enumerate(regionprops(label_im)):
        minr, minc, maxr, maxc = i.bbox
        roi = img[minr:maxr, minc:maxc]
        cv2.imwrite(output_path+ f"{num}.jpg", roi)


def hist(img):
    """ Calculates 3 channels color histogram with bins=256 
        
    ----------------------
    Parameters

        img: numpy.ndarray
            input image

    ----------------------
    Returns

        numpy.ndarray
            numpy array with shape (h, w, 3)

    """
    hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([img],[1],None,[256],[0,256])
    hist3 = cv2.calcHist([img],[2],None,[256],[0,256])
    

    return np.array([np.squeeze(hist1), np.squeeze(hist2), np.squeeze(hist3)])


def histDist(A, B):
    """ Calculates the distance between two images' histograms

    ----------------------
    Parameters

        A: numpy.ndarray
            base object
        
        B: numpy.ndarray
            cropped object to be verified
    
    ----------------------
    Returns

        float
            The distance of the compared histograms 
    
    """
    # Resize test image to base's size 
    base_shape = A.shape[:2]
    resized_B = cv2.resize(B, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_AREA)

    hist_A = hist(A)
    hist_B = hist(resized_B)
    res = cv2.compareHist(hist_A, hist_B, cv2.HISTCMP_BHATTACHARYYA)
    return res


def shape_distance(A, B):
    """ Calculates the proximity between two images' shape. Higher values
    corresponds to closer shapes. In this case, the image is the cropped 
    object

    ----------------------
    Parameters

        A: numpy.ndarray
            base object
        
        B: numpy.ndarray
            cropped object to be verified
    
    ----------------------
    Returns

        float
            Mean between the height and width ratio. A value 
            between 0 and 1 measuring how close the shapes are
    
    """
    A_shape = A.shape[:2]
    B_shape = B.shape[:2]
    height_ratio = 1 - abs((B_shape[0] - A_shape[0]) / A_shape[0])
    width_ratio = 1 - abs((B_shape[1] - A_shape[1]) / A_shape[1])
    mean = (height_ratio + width_ratio) / 2
    return mean


def shape_similarity(A, B):
    """ Rotate the image to compare it in different positions

    ----------------------
    Parameters

        A: numpy.ndarray
            Base image
        
        B: numpy.ndarray
            Test image
    
    ----------------------
    Returns

        Float
            Maximum value found compairing base and test images in different
            positions

    
    
    """
    aux = []
    for angle in range(-10, 10):
        rotated = rotate(B, angle=angle)
        res = shape_distance(A, B)
        aux.append(res)

    return (max(aux))


def verify_img(img):
    aux = []  # stores the histograms distances
    for base_img in base_imgs:
        # Compare the shape from test object with every base object
        # and if the difference is below 7%, calculate distance 
        # between histograms
        if (shape_distance(base_img, test_img) >= 0.9):

            # compare histograms and store into aux list
            res = histDist(base_img, test_img) 
            aux.append(res)

        # stores 1 for images with different shape
        else:
            aux.append(1)
            
    # From histogram distances, the minimum value's index is used to locate the name of
    # the object from the list of names
    minimum = min(aux)
    index = aux.index(minimum)
    match = objs[index]
    if minimum != 1 and minimum <= 0.55:
        print(f"Match --> {match:8}: Distance = {minimum}")
        actual_img.append(match)

    # missing = [element for element in objs if element not in actual_img]
    print(f"missing objects: {list(element for element in objs if element not in actual_img)}")