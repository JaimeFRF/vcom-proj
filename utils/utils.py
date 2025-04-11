import sys
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime
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

#show resulting images from pipeline
def show_images(imgs_data, size=(500, 500)):
    for i in imgs_data:
        cv2.imshow(f"final_{i['name']}", cv2.resize(i["image"], size))
    wait_for_exit()

def show_debug_images(imgs_data, gridFormat=False, gridImgSize=2, gridSaveFig=False, initialSize=(1200,700)):
    if gridFormat:
        imgs_count = len(imgs_data) # total number of images processed 
        debug_img_count = len(imgs_data[0]["debug"]) # number of debug images processed for each chessboard image
        
        fig, axs = plt.subplots(imgs_count, debug_img_count, figsize=(gridImgSize * debug_img_count, gridImgSize * imgs_count), num=f"Debug")
        
        for img_id, data in enumerate(imgs_data):
            for i, img in enumerate(data["debug"]):
                axs[img_id][i].imshow(cv2.cvtColor(img["image"], cv2.COLOR_BGR2RGB))
                axs[img_id][i].set_title(img["name"], size=3*gridImgSize)
                axs[img_id][i].axis("off")

        fig.tight_layout()

        if (gridSaveFig):
            fig_path = f"our_images/debug/debug_images_{datetime.now().strftime('%H_%M_%S')}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Debug images saved as {fig_path}")
        
        # Display the figure in a scrollable Qt window
        display_scrollable_figure(fig, title="Debug Images", initial_size=initialSize)

        plt.close(fig)


    else:
        for img in data["debug"]:
            cv2.imshow(img["name"], img["image"])
        wait_for_exit()

# helper function to show an image and add the original image name to the window title. Used for debug draw functions,
# this way, different images aren't overwriting the same named window throughout the pipeline (which would only make imshow windows visible for the last image in pipeline)
# so its less messy when debugging multiple chessboard images, we combine all images into a single window
def show_image_with_name(data, imageTitle, image):
    image_window_name = f"{imageTitle}_{data['name']}"
    data["debug"].append({"name": image_window_name, "image": image}) # add image to debug list
    # cv2.imshow(image_window_name, image)
    return data

#draw points for feature detection keypoints over original image
def draw_points(data, color=color_green, radius=3, thickness=2, imageTitle="Points", pointsFieldName="keypoints"):
    points = data["metadata"].get(pointsFieldName, None) # points data from previous function in pipeline
    if (points is None):
        raise ValueError(f"{pointsFieldName} data must be defined previously in pipeline, in order to draw points")
    
    img = data["image"].copy()

    for point in points:
        x, y = point.pt
        img = cv2.circle(img, (int(x), int(y)), radius, color, thickness)

    show_image_with_name(data, imageTitle, img)
    return data

#draw points for np array over original image
def draw_points_from_array(data, color=color_green, radius=3, thickness=2, imageTitle="Points", pointsFieldName="keypoints", makeColored=False):
    points = data["metadata"].get(pointsFieldName, None)
    if (points is None):
        raise ValueError(f"{pointsFieldName} data must be defined previously in pipeline, in order to draw points")
    
    if (points.shape[1] != 2 or len(points.shape) != 2):
        print(f"{pointsFieldName} data must be 2D array, ignoring drawing for {data['name']}")
        return data
    
    img = data["image"].copy()
    
    if (makeColored):
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    for point in points:
        [x, y] = point
        img = cv2.circle(img, (int(x), int(y)), radius, color, thickness)

    show_image_with_name(data, imageTitle, img)
    return data

# draw hough lines over original image
def draw_hough_lines(data, color=color_red, withText=False, textSize=1.5, houghLinesFieldName="lines", imageTitle="Hough Lines"):
    lines = data["metadata"].get(houghLinesFieldName, None) # lines data from previous function in pipeline
    if (lines is None):
        raise ValueError(f"{houghLinesFieldName} data must be defined previously in pipeline, in order to draw lines")
    
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
        
    show_image_with_name(data, imageTitle, img)
    return data

#draw chessboard calculated countours over original image
def draw_contours(data, imageTitle="Contours", color=color_green, thickness=3, contoursFieldName="board_contour"):
    contours = data["metadata"].get(contoursFieldName, None) # contours data from previous function in pipeline
    if (contours is None):
        raise ValueError("Contours data must be defined previously in pipeline, in order to draw contours")
    
    img = data["orig_img"].copy()
    img = cv2.drawContours(img, contours, -1, color, thickness)
    show_image_with_name(data, imageTitle, img)
    return data

# print a single image during pipeline
def show_current_image(data, imageTitle="current image", resizeAmount=1):
    resize_img = cv2.resize(data["image"], (0, 0), fx=resizeAmount, fy=resizeAmount)
    show_image_with_name(data, imageTitle, resize_img)
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

# set the current image in pipeline to the one stored in metadata
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

    show_image_with_name(data, imageTitle, train_with_box)

    return data

def draw_crosshair(data, color=color_white, thickness=2, size=20, imageTitle="Center point"):
    center_img = data["image"].copy()
    h, w = center_img.shape[:2]
    center = (w // 2, h // 2)
    cv2.drawMarker(center_img, center, color, cv2.MARKER_CROSS, size, thickness)
    show_image_with_name(data, imageTitle, center_img)

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

    show_image_with_name(data, imageTitle, img)
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
   
   
    show_image_with_name(data, imageTitle, img)
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

# make a matplotlib figure scrollable in a Qt window, so any figure size can be analyzed
def display_scrollable_figure(fig, title="Debug", initial_size=(700, 500)):

    # Ensure a QApplication instance exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create the main window
    window = QMainWindow()
    window.setWindowTitle(title)
    window.resize(*initial_size)

    # Create a widget and layout for the figure
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)

    # Create a canvas for the matplotlib figure
    canvas = FigureCanvas(fig)
    canvas.setMinimumSize(canvas.size())  # Prevent shrinking

    # Add the canvas to a scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidget(canvas)
    scroll_area.setWidgetResizable(False)  # Prevent resizing the canvas to fit

    # Add the scroll area to the layout
    layout.addWidget(scroll_area)
    window.setCentralWidget(central_widget)

    # Show the window
    # blocks until window is closed
    window.show()

    app.exec_()

# draw rectangles over the image, used for debugging
# rectangles: array of arrays [x1, y1, x2, y2] or [x, y, width, height] depending on withWidthHeight
# color: color of the rectangles
# thickness: thickness of the rectangle lines
# imageTitle: title of the image window
# fieldName: name of the field in metadata where the rectangles were stored
# withWidthHeight: if True, the rectangles are defined as [x, y, width, height], otherwise as [x1, y1, x2, y2]

def draw_rectangles(data, color=color_red, thickness=2, imageTitle="Rectangles", fieldName="rectangles", withWidthHeight=False, makeColored=False):
    rectangles = data["metadata"].get(fieldName, None)
    if (rectangles is None):
        raise ValueError(f"Field {fieldName} must be defined previously in pipeline, to print its value")

    img = data["image"].copy()
    if (makeColored):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    for rect in rectangles:
        if (withWidthHeight):
            [x, y, w, h] = map(int, np.round(rect))
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
        else:
            [x1, y1, x2, y2] = map(int, np.round(rect))
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    show_image_with_name(data, imageTitle, img)
    return data

def show_bounding_box_cuts(data, bbFieldName="bounding_boxes", imageTitle="BBCuts"):
    bboxes = data["metadata"].get(bbFieldName, None)
    if (bboxes is None):
        raise ValueError(f"Field {bbFieldName} must be defined previously in pipeline, to print its value")
    
    img = data["image"].copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        piece_img = img[y1:y2, x1:x2]

        show_image_with_name(data, f"{imageTitle}-bbox{i}", piece_img)
    return data

def download_current_image(data, path="temp/image"):
    final_path = f"{path}_{data['name']}"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    cv2.imwrite(final_path, data["image"])
    return data