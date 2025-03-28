import cv2
import math
import random
import numpy as np
import os

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

# save dimensions of an image into metadat to use later (e.g. for homography)
def save_image_dimensions_in_metadata(data, widthFieldTitle="width", heightFieldTitle="height"):
    data["metadata"][heightFieldTitle], data["metadata"][widthFieldTitle] = data["image"].shape[:2]
    return data

#save current image in metadata to use in prints later
def save_current_image_in_metadata(data, fieldName="image2"):
    data["metadata"][fieldName] = data["image"]
    return data

# set the current image to the one stored in metadata
def set_current_image(data, imageFieldName="image2"):
    data["image"] = data["metadata"][imageFieldName].copy()
    return data

# draw square box in current image, after applying a homography matrix to the box corners
# ex: used to draw small horse box in the warped img
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

#draw grid, the jupyter notebook way, has better results, but uses weird size?
def draw_grid(data, color=color_green, thickness=1, imageTitle="Grid", grid_cols=8, grid_rows=8, imgFieldName=None):
    img = None

    if (imgFieldName is not None):
        img = data["metadata"][imgFieldName].copy()
    else:
        img = data["image"].copy()

    size = round((465-44) / 8)
    #size = round((460-40) / 8)

    # Extract and save each tile
    for j in range(grid_cols):
        for i in range(grid_rows):
            # Get tile position
            x1, y1 = i * size, j * size
            x2, y2 = x1 + size, y1 + size

            # Draw rectangle on the original image for visualization
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow(imageTitle, img)
    data["image"] = img
    return data

# draw grid with lines for rows and columns, simpler way
def draw_grid_2(data, color=color_orange, thickness=1, imageTitle="Grid", grid_cols=8, grid_rows=8, imgFieldName=None):
    img = None

    if (imgFieldName is not None):
        img = data["metadata"][imgFieldName].copy()
    else:
        img = data["image"].copy()

    h, w = img.shape[:2]

    #draw vertical lines
    for j in range(1, grid_cols):
        line_x_index = j * w // grid_cols
        cv2.line(img, (line_x_index, 0), (line_x_index, h), color, thickness)
    
    #draw horizontal lines
    for i in range(1, grid_rows):
        line_y_index = i * h // grid_rows
        cv2.line(img, (0, line_y_index), (w, line_y_index), color, thickness)
   
   
    cv2.imshow(imageTitle, img)
    return data

def print_field_value(data, fieldName, withFieldName=False, withNewline=False):
    value = data["metadata"].get(fieldName, None)
    if (value is None):
        raise ValueError(f"Field {fieldName} must be defined previously in pipeline, to print its value")
    
    field_string = ""
    if (withFieldName):
        field_string += f"{fieldName}: "
    if (withNewline):
        field_string += "\n"
    print(f"{field_string}{value}")
    return data
