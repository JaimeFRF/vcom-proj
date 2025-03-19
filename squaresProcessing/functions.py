import cv2
import squaresProcessing.parameters as SquaresProcParams
import utils.utils as Utils
import math

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
    
# TODO: 
# - from the obtained lines in hough, filter in horizontal and vertical line groups
# - then pick the best equally spaced lines from each group
# - then calculate the intersections of those lines, to get the board corners
