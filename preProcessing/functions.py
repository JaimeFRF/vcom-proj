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

def table_segmentation(data, lower_color_bound=(150,100,50), upper_color_bound=(200, 150, 100)):

    hsv = cv2.cvtColor(data["orig_img"], cv2.COLOR_BGR2HSV)
    
    # assuming board is always in a table with orangeish colors, filter out the rest of the image that isn't part of the table
    lower_bound = np.array([0, 0, 100])
    upper_bound = np.array([180, 50, 255])
    
    # Create a mask for the board
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Apply the mask to the image
    data["image"] = cv2.bitwise_and(data["image"], data["image"], mask=mask)
    cv2.imshow("color segmented", data["image"])
    cv2.waitKey(0)
    
    return data