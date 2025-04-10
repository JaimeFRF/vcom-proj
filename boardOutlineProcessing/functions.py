import cv2
import boardOutlineProcessing.parameters as BoardOutlineProcParams
import utils.utils as Utils
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
def find_board_countour_and_corners(data, approxPolyDP_epsilon=0.05, min_perim=1000, max_perim=50000):
    find_countours(data)
    contours = data["metadata"].get("contours", None)

    if (contours is None):
        raise ValueError("No countours detected, so can't find board contour and corners")
    
    # filter resulting contours by area, since we are looking for a big square
    # reduces calculation times below, and removes small noise contours that are identified
    # TODO: is area more robust to calculate instead?
    filtered_countours = [cnt for cnt in contours if min_perim <= cv2.arcLength(cnt, True) <= max_perim]
    
    board_contour = None
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
                
    if board_contour is None:
        raise ValueError("No game board detected.")
    
    corners = board_contour.reshape(4, 2).astype(np.float32)

    data["metadata"]["board_contour"] = [board_contour]
    data["metadata"]["board_corners"] = corners

    return data

# given a set of corners, warp the image to a new perspective
def warp_image_from_board_corners(data, warp_width=500, warp_height=500, imgFieldName="orig_img"):
    corners = data["metadata"].get("board_corners", None)
    if (corners is None):
        raise ValueError("Board corners data must be defined previously in pipeline, in order to warp image")
    
    # define the destination points for the perspective transform
    dst_points = np.array([[0, 0], [warp_width-1, 0], [warp_width-1, warp_height-1], [0, warp_height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped = cv2.warpPerspective(data[imgFieldName].copy(), matrix, (warp_width, warp_height))
    
    data["image"] = warped
    
    return data

#helper fun
def helper(data):
    print("Image shape: ", data["image"].shape)
    data["metadata"]["tempimg"] = data["image"].copy() 
    #print(data)
    return data

def getimg(data):
    #print(data)
    data["image"] = data["metadata"]["tempimg"]
    return data

def binarize(data, threshold=127, max_value=255):
    # Convert to grayscale if image is color
    if len(data["image"].shape) > 2 and data["image"].shape[2] > 1:
        gray = cv2.cvtColor(data["image"], cv2.COLOR_BGR2GRAY)
    else:
        gray = data["image"].copy()
    
    # Convert to uint8 if needed
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    
    # If threshold is 0, use Otsu's method
    if threshold == 0:
        _, binary = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Regular binary thresholding
        _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
    
    data["image"] = binary
    return data



# Step 1: Find contours in the image
def find_contours_warp(data):
    # Ensure the image is binary (e.g., after thresholding)
    if "image" not in data or data["image"] is None:
        raise ValueError("No image data provided for contour detection.")
    
    # Find contours (assuming data["image"] is a binary image)
    contours, _ = cv2.findContours(data["image"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    data["metadata"]["contours"] = contours
    return data

# Step 2: Find the board contour and its corners
def find_board_contour_and_corners_warp(data, approxPolyDP_epsilon=0.05, min_perim=1000, max_perim=5000):
    if "metadata" not in data:
        data["metadata"] = {}
    
    data = find_contours_warp(data)
    contours = data["metadata"].get("contours", None)

    if contours is None or len(contours) == 0:
        raise ValueError("No contours detected, so can't find board contour and corners.")
    
    filtered_contours = [cnt for cnt in contours if min_perim <= cv2.arcLength(cnt, True) <= max_perim]
    if not filtered_contours:
        raise ValueError("No contours found within perimeter range.")
    
    board_contours = []  # Store all valid quadrilaterals
    for contour in filtered_contours:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, approxPolyDP_epsilon * peri, True)
        
        if len(approx) == 4:  # Keep all quadrilaterals
            board_contours.append(approx)
    
    if not board_contours:
        raise ValueError("No valid board contours detected (no quadrilaterals found).")
    
    # Store all contours (not just one)
    data["metadata"]["board_contour"] = board_contours
    # Optionally store corners for all contours
    data["metadata"]["board_corners"] = [cnt.reshape(4, 2).astype(np.float32) for cnt in board_contours]
    
    print("Number of board contours detected:", len(board_contours))
    return data

