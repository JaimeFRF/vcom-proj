import cv2

#Blur

#reduce noise with smoothing, but lose edges quality
def gaussian(data, ksize =(5, 5)):
    data["image"] = cv2.GaussianBlur(data["image"], ksize, 0)
    return data


def clahe(data, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    data["image"] = clahe.apply(data["image"])
    return data



def median(data, ksize = 5):
    data["image"] = cv2.medianBlur(data["image"], ksize)
    return data

# reduce noise with smoothing, but keep edges sharp
#source: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
def bilateral(data, d = 15, sigmaColor = 75, sigmaSpace = 75):
    data["image"] = cv2.bilateralFilter(
        data["image"], 
        d, # pixel neighborhood used during filtering, aka kernel size
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
