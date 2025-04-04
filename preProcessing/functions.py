import cv2
import numpy as np

#reduce noise with smoothing, but lose edges quality
def gaussian(data, ksize =(5, 5)):
    data["image"] = cv2.GaussianBlur(data["image"], ksize, 0)
    return data

def clahe(data, clipLimit=2.0, tileGridSize=(10, 10)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    data["image"] = clahe.apply(data["image"])
    return data

def median(data, ksize = 5):
    data["image"] = cv2.medianBlur(data["image"], ksize)
    return data

# reduce noise with smoothing, but keep edges sharp
#source: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
def bilateral(data, ksize = 5, sigmaColor = 75, sigmaSpace = 75):
    data["image"] = cv2.bilateralFilter(
        data["image"], 
        ksize, # pixel neighborhood used during filtering, aka kernel size
        sigmaColor, #larger value -> farther colors within the pixel neighborhood (based on sigmaSpace) will be mixed together
        sigmaSpace) #larger value -> farther pixels will influence each other as long as their colors are close enough (based on sigmaColor)
    return data

# Binarization

def convert_to_gray(data):
    data["image"] = cv2.cvtColor(data["image"], cv2.COLOR_BGR2GRAY)
    return data

# Normalization
def normalize(data, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
    data["image"] = cv2.normalize(data["image"], None, alpha, beta, norm_type)
    return data

# Equalize Histogram for main pipeline image
def equalizeHist(data):
    data["image"] = cv2.equalizeHist(data["image"])
    return data

# calculate a convex hull mask for a certain color range, from the current image
# Ex: for identifying the table region
def color_mask(data, lower_color_bound=(10, 50, 60), upper_color_bound=(20, 255, 255), mask_edges_erosion=10, colorMaskFieldTitle="color_mask"):

    hsv = cv2.cvtColor(data["image"], cv2.COLOR_BGR2HSV)
    
    # assuming board is always in a table with orangeish colors, filter out the rest of the image that isn't part of the table
    lower_bound = np.array(list(lower_color_bound))
    upper_bound = np.array(list(upper_color_bound))
    
    # Create a mask for the board
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    _,thresh = cv2.threshold(mask, 40, 255, 0)

    # opening over the thresholded image, to remove residual noise from mask
    kernel = np.ones(11, np.uint8)
    eroded_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

    contours, hierarchy = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if (len(contours) == 0):
        raise ValueError("No contours detected, so can't find board contour and corners")
    
    # grab the biggest contour, which should be the table
    biggest_contour = max(contours, key = cv2.contourArea)

    # find the convex hull of the contour, to compensate for any brightness noise inside the table region
    hull = cv2.convexHull(biggest_contour) 

    #create mask for the convex hull
    mask = np.zeros((data["image"].shape[0], data["image"].shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, (255,255,255), -1)
    
    # reduce mask in edges, with erosion, so when we apply it later, it also excludes the table edges (from which we calculated this mask)
    kernel = np.ones(5, np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=mask_edges_erosion)

    # result = cv2.add(table_section, solid_color_section)
    data["metadata"][colorMaskFieldTitle] = eroded_mask

    return data

#apply a single channel mask to a greyscale image
def apply_mask(data, maskFieldTitle="color_mask", imageFieldTitle=None):
    mask = data["metadata"].get(maskFieldTitle, None)
    if (mask is None):
        raise ValueError("No mask found in metadata")

    data["image"] = cv2.bitwise_and(data["image"], data["image"], mask=mask)
    return data