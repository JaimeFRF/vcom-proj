
# bilateral filter
bilateral_ksize = 9 # larger filter for more smoothing
#since black and white chess colors are very distinct, we can use high sigmaValues to smooth noise, without losing edges definition
bilateral_sigmaColor = 255 # larger value -> farther colors within the pixel neighborhood (based on sigmaSpace) will be mixed together

#log transform
log_constant = 30.0

#clahe filter
clipLimit = 3.0
tileGridSize = (8, 8)

# bounding box calculations
bbVertFactor = 45000

#image contrast adjust
gamma = 0.38
cutoff= 256

#find piece contours in bboxes
white_lower_bound = (15, 32, 140)
white_upper_bound = (30, 180, 255)

black_lower_bound = (0, 0, 0)
black_upper_bound = (180, 70, 120)

white_edges_kernel = (4,4)
black_edges_kernel = (9,9)

piece_min_area = 2000
piece_max_center_dist = 0.3