import cv2
import math
import numpy as np

## Module for functions related to identifying the board layout, and warping it to have a straight grid perspective

# detect edges in image
def canny(data, low=100, high=200):
    data["image"] = cv2.Canny(
        data["image"],
        low, #threshold to be considered weak edge
        high, #threshold to be considered strong edge,
    )
    return data

# detect lines, from edges of image
def hough_lines(data, rho=1, theta=math.pi / 180, votes=150):
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

# erode edges to make them less visible
# erodes away the boundaries of foreground object
def erode_edges(data, ksize=(3, 3), iterations=1):
    kernel = np.ones(ksize, np.uint8)

    data["image"] = cv2.erode(data["image"], kernel, iterations=iterations)
    return data

# dilate edges to make them more visible
# makes objects more visible and fills in small holes in objects. Lines appear thicker, and filled shapes appear larger
def dilate_edges(data, ksize=(3, 3), iterations=1):
    kernel = np.ones(ksize, np.uint8)

    data["image"] = cv2.dilate(data["image"], kernel, iterations=iterations)
    return data

# closing image - dilation followed by erosion, to remove noise in image
# useful in closing small holes inside the foreground objects, or small black points on the object.
# source: https://www.geeksforgeeks.org/python-opencv-morphological-operations/
#         https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def closing(data, ksize=(5, 5), iterations=1):
    kernel = np.ones(ksize, np.uint8)

    data["image"] = cv2.morphologyEx(data["image"], cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return data

# opening image - erosion followed by dilation, to remove noise in image
# source: https://www.geeksforgeeks.org/python-opencv-morphological-operations/
#         https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def opening(data, ksize=(5, 5), iterations=1):
    kernel = np.ones(ksize, np.uint8)
    data["image"] = cv2.morphologyEx(data["image"], cv2.MORPH_OPEN, kernel, iterations=iterations)
    return data

# detect contours in image - joining all the points along the boundary of an image that are having the same intensity
# "works best on binary images, so we should first apply thresholding techniques, Sobel edges, etc."
# source: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
#         https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=cv2.findcontours#cv2.findContours
def find_countours(data):
    contours, _ = cv2.findContours(data["image"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    data["metadata"]["contours"] = contours
    return data

# find the board contour and its corners
def find_board_countour_and_corners(data, approxPolyDP_epsilon=0.05, min_perim=1000, max_perim=50000, boardContourFieldName="board_contour", boardContourRawFieldName="board_contour_raw", boardCornersFieldName="board_corners"):
    find_countours(data)
    contours = data["metadata"].get("contours", None)

    if (contours is None):
        raise ValueError("No countours detected, so can't find board contour and corners")
    
    # filter resulting contours by area, since we are looking for a big square
    # reduces calculation times below, and removes small noise contours that are identified
    # TODO: is area more robust to calculate instead?
    filtered_countours = [cnt for cnt in contours if min_perim <= cv2.arcLength(cnt, True) <= max_perim]
    
    board_contour = None # final contour of the board, after applying hull, and approxPolyDP
    board_contour_raw = None  # raw final contour, without applying hull or approxPolyDP # useful for masking later on on the pipeline
    max_area = 0
    i = 0
    for contour in filtered_countours:
        # Get convex hull because it works better
        # source https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        hull = cv2.convexHull(contour) # convex hull surrounding contour
        peri = cv2.arcLength(hull, True) # perimeter of the contour
        approx = cv2.approxPolyDP(hull, approxPolyDP_epsilon * peri, True)  # simplify countour # higher epsilon -> less vertices -> better board shape approximation
        
        #TODO: conseguir separar background de foreground, para poder filtrar depois o contour especifico da board
        #TODO:filtrar contours by solidity -> meansure of how much the contour occupies the convex hull, to guarantee is polygon shaped

        # if the final contour is a polygon with 4 sides (a square in perspective)
        if len(approx) == 4:
            i += 1
            # print("Contour" + str(i))
            area = cv2.contourArea(contour) # calculate area of contour
            if area > max_area:
                max_area = area
                board_contour = approx # get the contour with the biggest area, cause the biggest square in the image is the board
                board_contour_raw = contour # save the selected contour, to be used for masking later on

    if board_contour is None:
        raise ValueError("No game board detected.")
    
    corners = board_contour.reshape(4, 2).astype(np.float32)

    data["metadata"][boardContourFieldName] = [board_contour]
    data["metadata"][boardContourRawFieldName] = [board_contour_raw]
    data["metadata"][boardCornersFieldName] = corners

    return data

# given a set of corners, warp the image to a new perspective
def warp_image_from_board_corners(data, warp_width=500, warp_height=500, warpMatrixFieldName="warp_matrix", imgFieldName="orig_img", boardCornersFieldName="board_corners"):
    corners = data["metadata"].get(boardCornersFieldName, None)
    if (corners is None):
        raise ValueError("Board corners data must be defined previously in pipeline, in order to warp image")
    
    # define the destination points for the perspective transform
    dst_points = np.array([[0, 0], [warp_width-1, 0], [warp_width-1, warp_height-1], [0, warp_height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped = cv2.warpPerspective(data[imgFieldName].copy(), matrix, (warp_width, warp_height))
    
    data["image"] = warped
    data["metadata"][warpMatrixFieldName] = matrix
    
    return data