import math

#canny edge detection
canny_low_threshold = 50
canny_high_threshold = 150

# dilate edges
dilate_ksize = (3, 3)
dilate_iterations = 1

#closing image
closing_ksize = (7, 7)
closing_iterations = 3

#opening image
opening_ksize = (3, 3)
opening_iterations = 1

#find_board_contour_and_corners
approxPolyDP_epsilon = 0.05 # higher epsilon -> less vertices -> better board shape approximation to a square

# warp image
warp_width = warp_height = 500

#hough lines
hough_rho = 1 # resolution in pixels of the Hough grid
hough_theta = math.pi / 180.0 # angle resolution in radians
hough_votes = 200 # min number of points aligned, for valid line
