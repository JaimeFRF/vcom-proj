import cv2
import math
import random
import numpy as np
#Colors
color_red = (0,0,255)
color_green = (0,255,0)
color_blue = (255,0,0)
color_white = (255,255,255)
color_orange = (0,165,255)
color_yellow = (51,255,255)

def wait_for_exit():
    while cv2.waitKey(1) != ord("q"):
        pass
    cv2.destroyAllWindows()

def show_images(imgs_data, size=(500, 500)):
    for i in imgs_data:
        cv2.imshow(i["name"], cv2.resize(i["image"], size))

    wait_for_exit()

def draw_hough_lines(data, color=color_red, withText=False, textSize=1.5):
    lines = data["metadata"].get("lines", None) # lines data from previous function in pipeline
    if (lines is None):
        raise ValueError("Lines data must be defined previously in pipeline, in order to draw lines")
    
    img = data["orig_img"].copy()

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 4000*(-b)), int(y0 + 4000*(a)))
        pt2 = (int(x0 - 4000*(-b)), int(y0 - 4000*(a)))

        # Draw the line
        img = cv2.line(img, pt1, pt2, color, 3)

        #draw text to describe the line properties
        if (withText):
            pt1_text = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2_text = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            text_pos = (int((pt1_text[0] + pt2_text[0]) /2), int((pt1_text[1] + pt2_text[1]) /2) + random.randint(5,50))
            cv2.putText(img, f"r:{rho}, ang:{theta}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, textSize, color, 1, cv2.LINE_AA)
        
    cv2.imshow("Hough Lines", img)
    return data

def draw_contours(data, imageTitle="Contours", color=color_green, thickness=3, fieldName="contours"):
    contours = data["metadata"].get(fieldName, None) # contours data from previous function in pipeline
    if (contours is None):
        raise ValueError("Contours data must be defined previously in pipeline, in order to draw contours")
    
    img = data["orig_img"].copy()
    img = cv2.drawContours(img, contours, -1, color, thickness)
    cv2.imshow(imageTitle, img)
    return data

# print a single image during pipeline
def show_current_image(data, imageTitle="current image", resizeAmount=1):
    resize_img = cv2.resize(data["image"], (0, 0), fx=resizeAmount, fy=resizeAmount)
    cv2.imshow(imageTitle, resize_img)
    return data

# read an image from a path, during the pipeline, and add it to the data dictionary
def read_other_image(data, path, format=cv2.IMREAD_GRAYSCALE, imageFieldName="newImage"):
    data["metadata"][imageFieldName] = cv2.imread(path, format)
    return data

def save_image_dimensions_in_metadata(data, widthFieldTitle="width", heightFieldTitle="height"):
    data["metadata"][heightFieldTitle], data["metadata"][widthFieldTitle] = data["image"].shape[:2]
    return data

def save_current_image_in_metadata(data, fieldName="image2"):
    data["metadata"][fieldName] = data["image"]
    return data

def draw_perspective_transformed_points(data, color=color_yellow, linesThickness=2, widthTitle="width", heightTitle="height", homographyTitle="homography", imageTitle="Train Image with Object Outline"):

    width = data["metadata"].get(widthTitle, None)
    height = data["metadata"].get(heightTitle, None)
    M = data["metadata"].get(homographyTitle, None)

    if (M is None or width is None or height is None):
        raise ValueError("Homography matrix, width and height must be defined previously in pipeline, to draw perspective transform image")

    corners = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)

    # Transform corners to train image coordinates
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # Create a copy of train image for drawing
    train_with_box = data["image"].copy()

    # Draw lines connecting the transformed corners
    transformed_corners = np.int32(transformed_corners)
    cv2.polylines(train_with_box, [transformed_corners], True, color, linesThickness)

    cv2.imshow(imageTitle, train_with_box)

    return data

def draw_crosshair(data, color=color_white, thickness=2, size=20, imageTitle="Center point"):
    center_img = data["image"].copy()
    h, w = center_img.shape[:2]
    center = (w // 2, h // 2)
    cv2.drawMarker(center_img, center, color, cv2.MARKER_CROSS, size, thickness)
    cv2.imshow(imageTitle, center_img)

    return data

