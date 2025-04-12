
# bilateral filter
bilateral_ksize = 9 # larger filter for more smoothing
#since black and white chess colors are very distinct, we can use high sigmaValues to smooth noise, without losing edges definition
bilateral_sigmaColor = 255 # larger value -> farther colors within the pixel neighborhood (based on sigmaSpace) will be mixed together

#clahe filter
clipLimit = 3.0
tileGridSize = (8, 8)

# bounding box calculations
bbVertFactor = 30000

#image contrast adjust
gamma = 0.35
cutoff= 256

#find piece contours in bboxes
white_lower_bound = (15, 85, 160)
white_upper_bound = (32, 200, 255)
black_lower_bound = (0, 0, 0)
black_upper_bound = (180, 70, 120)
white_edges_erosion = 4
black_edges_erosion = 7
piece_min_area = 2000
piece_max_center_dist = 0.3