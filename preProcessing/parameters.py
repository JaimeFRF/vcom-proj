import math

#gaussian filter
gaussian_ksize = (5, 5)

#bilateral filter
bilateral_d = 15
#since black and white chess colors are very distinct, we can use high sigmaValues to smooth noise, without losing edges definition
bilateral_sigmaColor = 200
bilateral_sigmaSpace = 200