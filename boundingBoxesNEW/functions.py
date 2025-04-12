import math
import cv2
import utils.utils as Utils
import numpy as np
import singleSquaresProcessing.parameters as singleParams

# cornerCutSize -> size of the corners previously cut from the image
# resultsMatrixFieldName -> name of the field in metadata where the results matrix were stored
# warpMatrixFieldName -> name of the field in metadata where the warp matrix used to obtain the straight grid layout previously were stored
# rotationMatrixFieldName -> name of the field in metadata where the rotation matrix calculated from little horse was stored
# cornersFieldName -> name of the field in metadata where the occupied squares corners will be stored
def get_occupied_squares_corners(data, boardCornerCutSize=0, resultsMatrixFieldName="chessboard_matrix", warpMatrixFieldName="warp_matrix", rotationMatrixFieldName="rotation_matrix", cornersFieldName="occupied_squares_corners"):
    results_matrix = data["metadata"].get(resultsMatrixFieldName, None)
    warp_matrix = data["metadata"].get(warpMatrixFieldName, None)
    rotation_matrix_2d = data["metadata"].get(rotationMatrixFieldName, None)
    
    if results_matrix is None:
        raise ValueError("Results matrix data must be defined previously in pipeline, in order to get occupied squares corners")
    if warp_matrix is None:
        raise ValueError("Warp matrix data must be defined previously in pipeline, in order to get occupied squares corners")
    if rotation_matrix_2d is None: 
        raise ValueError("Rotation matrix data must be defined previously in pipeline, in order to get occupied squares corners")

    img_h, img_w = data["image"].shape[:2]
    square_h, square_w = (img_h - boardCornerCutSize * 2) // 8, (img_w - boardCornerCutSize * 2) // 8

    #3x3 matrix for 2d affine
    rotation_matrix_3d = np.vstack([rotation_matrix_2d, [0,0,1]])

    inv_rotation_matrix = np.linalg.inv(rotation_matrix_3d) # used for un-rotating the image
    inv_warp_matrix = np.linalg.inv(warp_matrix) # used for un-warping the image

    final_matrix = inv_warp_matrix @ inv_rotation_matrix #final matrix to apply, that inverts both transformations # first undo the rotation, then undo the warping

    #calculate corners of occupied squares
    final_squares_corners = []
    for row in range(8):
        for col in range(8):
            if results_matrix[row][col] != 0:

                #calculate the corners of the occupied square
                x1 = boardCornerCutSize + col * square_w
                y1 = boardCornerCutSize + row * square_h
                x2 = x1 + square_w
                y2 = y1 + square_h

                #add additional dimension for matrix multiplication (top_left, top_right, bottom_left, bottom_right)
                final_squares_corners.extend([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]])
    
    if not final_squares_corners:
        data["metadata"][cornersFieldName] = np.empty((0, 2))
        data["metadata"]["debug_corners_warped_img"] = np.empty((0, 2))
        print("No occupied squares found in results_matrix; cannot compute corners.")
        return data

    final_squares_corners = np.array(final_squares_corners).T # shape (3,N)

    #normalize back to 2 dimensions points
    transformed_final_squares_corners = final_matrix @ final_squares_corners
    transformed_final_squares_corners = (transformed_final_squares_corners[:2, :] / transformed_final_squares_corners[2, :]).T # shape(N,2)

    data["metadata"][cornersFieldName] = transformed_final_squares_corners
    data["metadata"]["debug_corners_warped_img"] = (final_squares_corners[:2, :] / final_squares_corners[2, :]).T
    return data

# given a set of occupied tile corners, calculate the initial static bounding boxes for each of those tile's pieces (based on static dimensions)
# cornersFieldName: name of the field in metadata where the occupied squares corners were stored
# bbFieldName: name of the field in metadata where the static bounding boxes will be stored
def get_pieces_static_bounding_boxes(data, cornersFieldName="occupied_squares_corners", bbFieldName="bounding_boxes", bbVertFactor=2):
    corners = data["metadata"].get(cornersFieldName, None)
    if (corners is None):
        raise ValueError("Occupied squares corners data must be defined previously in pipeline, in order to get pieces static bounding boxes")
    
    if (corners.shape[0] == 0):
        print("No occupied squares found in corners; cannot compute bounding boxes.")
        data["metadata"][bbFieldName] = np.empty((0, 4))
        return data
    
    # calculate the bounding boxes for each occupied square
    bounding_boxes = []
    num_tiles = len(corners) // 4
    
    #for each square
    for i in range(num_tiles):

        # calculate x and y maxs/min for the tile
        tile_corners = corners[i*4:(i+1)*4]
        min_x = np.min(tile_corners[:, 0])
        max_x = np.max(tile_corners[:, 0])
        min_y = np.min(tile_corners[:, 1])
        max_y = np.max(tile_corners[:, 1])
        tile_height = max_y - min_y
        
        #calculate the bounding box
        bb_x1 = min_x
        bb_y1 = min_y - (bbVertFactor / tile_height)  # if photo is taken more from the side, tile height is less, and piece height is more, so inverse relation between tile height and piece height!
        bb_x2 = max_x
        bb_y2 = max_y

        bounding_boxes.append(
            [
                int(round(bb_x1)), 
                int(round(bb_y1)), 
                int(round(bb_x2)), 
                int(round(bb_y2))
            ])
    
    data["metadata"][bbFieldName] = bounding_boxes
    return data

# find contours of pieces in each boundin box
# bbFieldName: name of the field in metadata where the static bounding boxes were stored
# refinedBbFieldName: name of the field in metadata where the refined boundinb boxes will be stored
# whiteLowerBound: lower bound for the white piece color in HSV
# whiteUpperBound: upper bound for the white piece color in HSV
# blackLowerBound: lower bound for the black piece color in HSV
# blackUpperBound: upper bound for the black piece color in HSV
# whiteEdgesErosion: size of the kernel for the white piece edges erosion
# blackEdgesErosion: size of the kernel for the black piece edges erosion
# resultsMatrixFieldName: name of the field in metadata where the results matrix was stored

def refine_bounding_boxes(data, bbFieldName="bounding_boxes", refinedBbFieldName="refined_bounding_boxes", whiteLowerBound=(20, 60, 50), whiteUpperBound=(25, 150, 255), blackLowerBound=(10, 50, 60), blackUpperBound=(20, 255, 255), whiteEdgesErosion=5, blackEdgesErosion=5, resultsMatrixFieldName="chessboard_matrix"):
    bboxes = data["metadata"].get(bbFieldName, None)
    if (bboxes is None):
        raise ValueError("Bounding boxes data must be defined previously in pipeline, in order to find piece contours")
    if (len(bboxes) == 0):
        print("No bounding boxes found; cannot find piece contours.")
        data["metadata"][refinedBbFieldName] = np.array([])
        return data
    
    results_matrix = data["metadata"].get(resultsMatrixFieldName, None)
    if (results_matrix is None):
        raise ValueError(f"{resultsMatrixFieldName} data must be defined previously in pipeline, in order to find piece contours")
    
    hsv_image = cv2.cvtColor(data["image"], cv2.COLOR_BGR2HSV)

    #remove all 0s and flatten results matrix, to get only the color of the occupied squares
    occupied_squares = results_matrix[results_matrix != 0]
    print("occupied_squares", occupied_squares)
    refined_bboxes = []

    # cv2.imshow("og_img", data["image"])

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        # get the bounding box of the piece
        piece_bbox = data["image"][y1:y2, x1:x2]
        # cv2.imshow("piece_bbox", piece_bbox)
        
        piece_mask = None
        refined_bbox = None

        # if occupying piece is white, use white color mask
        if (occupied_squares[i] == 1):
            piece_mask = _get_color_mask(piece_bbox, lowerBound=whiteLowerBound, upperBound=whiteUpperBound, kernelFactor=whiteEdgesErosion, iterFactor=5, morphologyType=cv2.MORPH_OPEN)
        
        # if occupying piece is black, use black color mask
        else:
            piece_mask = _get_color_mask(piece_bbox, lowerBound=blackLowerBound, upperBound=blackUpperBound, kernelFactor=blackEdgesErosion, iterFactor=5, morphologyType=cv2.MORPH_CLOSE)
        
        # cv2.imshow("piece_mask", piece_mask)
        # cv2.waitKey(0)
        
        #find best contour from image mask mask
        best_contour = _find_best_contour(piece_mask)

        if (best_contour is None):
            # print("No good contours detected, using previous bbox")
            refined_bboxes.append(bbox)
            continue
    
        x,y,w,h = cv2.boundingRect(best_contour) # get the bounding box of the biggest contour
        refined_bbox = [x, y, x + w, y + h]
        refined_bboxes.append(refined_bbox)

    data["metadata"][refinedBbFieldName] = refined_bboxes

    return data

def _get_color_mask(image, lowerBound=(0,0,0), upperBound=(128,128,128), kernelFactor=5, iterFactor=3, morphologyType=cv2.MORPH_OPEN):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_image, np.array(list(lowerBound)), np.array(list(upperBound)))
    _,thresh = cv2.threshold(mask, 0, 255, 0)

    # cv2.imshow("thresh", thresh)
    kernel = np.ones(kernelFactor, np.uint8)
    thresh = cv2.morphologyEx(thresh, morphologyType, kernel, iterations=iterFactor) # we open because complete white pieces are easy to detect, so we just want to remove the noise
    # cv2.imshow(f"{morphologyType}_thresh", open_white_thresh)
    # cv2.waitKey(0)
    return thresh

# find the best contour from the piece mask
# min_area: minimum area of the contour to be considered
# max_center_dist: maximum distance from the center of the image to the center of the contour (0-0.5)
def _find_best_contour(piece_mask, min_area=4000, max_center_dist=0.25):
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    h, w = piece_mask.shape[:2]
    center = (w // 2, h // 2)
    best_contour = None
    best_area = float(0)

    for cnt in contours:
        cnt_bbox = cv2.boundingRect(cnt)
        x, y, bw, bh = cnt_bbox
        cx, cy = x + bw/2, y + bh/2
        area = cv2.contourArea(cnt)
        dist = np.linalg.norm(center - np.array([cx, cy])) / max(w, h) # calculate distance from center of image to center of contour
        # print("area-dist", area, dist)
        temp_mask = cv2.cvtColor(piece_mask, cv2.COLOR_GRAY2BGR)
        temp_mask = cv2.drawContours(temp_mask, [cnt], -1, (0, 0, 255), 3)
        # cv2.imshow("contour", temp_mask)
        # cv2.waitKey(0)

        if dist < max_center_dist and area > min_area and area > best_area:
            best_area = area
            best_contour = cnt

    if best_contour is None:
        # print("No contours found")
        return None
    
    # draw the biggest contour on the image
    # piece_mask= cv2.drawContours(piece_mask, [cnt], -1, (0, 255, 0), 3)
    # cv2.imshow("biggest_contour", piece_mask)
    # cv2.waitKey(0)