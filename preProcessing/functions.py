import cv2

#Blur

def gaussian(img, ksize = (5, 5)):
    return (img[0], cv2.GaussianBlur(img[1], ksize, 0))

def median(img, ksize = 5):
    return (img[0], cv2.medianBlur(img[1], ksize))

def bilateral(img, d = 15, sigmaColor = 75, sigmaSpace = 75):
    return (img[0], cv2.bilateralFilter(img[1], d, sigmaColor, sigmaSpace)) 

# Binarization

def convert_to_gray(img):
    return (img[0], cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY))

# Normalization

def normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
    return (img[0], cv2.normalize(img[1], None, alpha, beta, norm_type))

# Equalize Histogram

def equalizeHist(img):
    return (img[0], cv2.equalizeHist(img[1]))