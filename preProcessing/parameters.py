import math

#gaussian filter
gaussian_ksize = (5, 5)

#bilateral filter
bilateral_ksize = 9 # larger filter for more smoothing
#since black and white chess colors are very distinct, we can use high sigmaValues to smooth noise, without losing edges definition
bilateral_sigmaColor = 100 # larger value -> farther colors within the pixel neighborhood (based on sigmaSpace) will be mixed together
bilateral_sigmaSpace = 200 # irrelevant, defined by the ksize value

#clahe filter
#NOTE: increases contrast of wood grain, aka makes noise more visible, not good for our use case
# source: https://stackoverflow.com/questions/38504864/opencv-clahe-parameters-explanation
clahe_clipLimit = 2.0
clahe_tileGridSize = (10, 10) #Input image will be divided into equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.

