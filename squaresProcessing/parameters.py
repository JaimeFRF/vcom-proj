import math

#canny edge detection
canny_low_threshold = 150
canny_high_threshold = 200

#hough lines
hough_rho = 1 # resolution in pixels of the Hough grid
hough_theta = math.pi / 180.0 # angle resolution in radians
hough_votes = 200 # min number of points aligned, for valid line
