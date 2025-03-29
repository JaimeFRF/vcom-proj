import cv2
import squaresProcessing.parameters as SquaresProcParams
import utils.utils as Utils
import math
import numpy as np

## Module for functions related to identifying the board layout, and warping it to have a straight grid perspective

# detect edges in image
def canny(data, low=100, high=200):
    data["image"] = cv2.Canny(
        data["image"],
        low, #threshold to be considered weak edge
        high #threshold to be considered strong edge
    )
    return data

# detect lines, from edges of image
def hough_lines(data, rho=1, theta=math.pi / 180, votes=50):
    lines = cv2.HoughLines(
        data["image"],
        rho, # resolution in pixels of the Hough grid -> higher value more precise distances of lines
        theta, # resolution in radians of the Hough grid -> higher value more precise angles of lines 
        votes # min points to form a line -> higher number, more prominent lines only
    )

    if lines is None:
        raise ValueError("No lines detected")
    
    data["metadata"]["lines"] = lines # new key in the dictionary with the lines obtained, that can be used by later functions in the pipeline
    return data
    
# dilate edges to make them more visible
def dilate_edges(data, ksize=(5, 5), iterations=1):
    kernel = np.ones(ksize, np.uint8)

    data["image"] = cv2.dilate(data["image"], kernel, iterations=iterations)
    return data

# closing image - dilation followed by erosion, to remove noise in image
# source: https://www.geeksforgeeks.org/python-opencv-morphological-operations/
def closing(data, ksize=(5, 5), iterations=3):
    kernel = np.ones(ksize, np.uint8)

    data["image"] = cv2.morphologyEx(data["image"], cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return data

# detect contours in image - joining all the points along the boundary of an image that are having the same intensity
# "works best on binary images, so we should first apply thresholding techniques, Sobel edges, etc."
# source: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
def find_countours(data):
    contours, _ = cv2.findContours(data["image"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    data["metadata"]["contours"] = contours
    return data

# find the board contour and its corners
def find_board_countour_and_corners(data, approxPolyDP_epsilon=0.05):
    find_countours(data)
    contours = data["metadata"].get("contours", None)

    if (contours is None):
        raise ValueError("No countours detected, so can't find board contour and corners")
    
    board_contour = None
    max_area = 0
    i = 0
    for contour in contours:
        # Get convex hull because it works better
        # source https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        hull = cv2.convexHull(contour) # convex hull surrounding contour
        peri = cv2.arcLength(hull, True) # perimeter of the contour
        approx = cv2.approxPolyDP(hull, approxPolyDP_epsilon * peri, True)  # simplify countour # higher epsilon -> less vertices -> better board shape approximation
        
        # if the final contour is a polygon with 4 sides (a square in perspective)
        if len(approx) == 4:
            i += 1
            # print("Contour" + str(i))
            area = cv2.contourArea(contour) # calculate area of contour
            if area > max_area:
                max_area = area
                board_contour = approx # get the contour with the biggest area, cause the biggest square in the image is the board
                
    if board_contour is None:
        raise ValueError("No game board detected.")
    
    corners = board_contour.reshape(4, 2).astype(np.float32)

    data["metadata"]["board_contour"] = [board_contour]
    data["metadata"]["board_corners"] = corners

    return data

# given a set of corners, warp the image to a new perspective
def warp_image_from_board_corners(data, warp_width=500, warp_height=500):
    corners = data["metadata"].get("board_corners", None)
    if (corners is None):
        raise ValueError("Board corners data must be defined previously in pipeline, in order to warp image")
    
    # define the destination points for the perspective transform
    dst_points = np.array([[0, 0], [warp_width-1, 0], [warp_width-1, warp_height-1], [0, warp_height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped = cv2.warpPerspective(data["orig_img"].copy(), matrix, (warp_width, warp_height))
    
    data["image"] = warped
    
    return data