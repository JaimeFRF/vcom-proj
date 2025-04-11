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
        print("No occupied squares found in results_matrix; cannot compute corners.")
        return data

    final_squares_corners = np.array(final_squares_corners).T # shape (3,N)

    #normalize back to 2 dimensions points
    transformed_final_squares_corners = final_matrix @ final_squares_corners
    transformed_final_squares_corners = (transformed_final_squares_corners[:2, :] / transformed_final_squares_corners[2, :]).T # shape(N,2)

    data["metadata"][cornersFieldName] = transformed_final_squares_corners
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
def find_piece_contours_in_bounding_boxes(data, bbFieldName="bounding_boxes", pieceContoursFieldName="piece_contours", white_lower_bound=(20, 60, 50), white_upper_bound=(25, 150, 255), black_lower_bound=(10, 50, 60), black_upper_bound=(20, 255, 255),  mask_edges_erosion=10):
    bboxes = data["metadata"].get(bbFieldName, None)
    if (bboxes is None):
        raise ValueError("Bounding boxes data must be defined previously in pipeline, in order to find piece contours")
    if (len(bboxes) == 0):
        print("No bounding boxes found; cannot find piece contours.")
        data["metadata"][pieceContoursFieldName] = []
        return data
    
    hsv_image = cv2.cvtColor(data["image"], cv2.COLOR_BGR2HSV)

    # Create a mask for the white piece color
    white_mask = cv2.inRange(hsv_image, np.array(list(white_lower_bound)), np.array(list(white_upper_bound)))
    # Create a mask for the black piece color
    black_mask = cv2.inRange(hsv_image, np.array(list(black_lower_bound)), np.array(list(black_upper_bound)))
    
    #merge both masks
    complete_mask = cv2.bitwise_or(white_mask, black_mask)

    data["metadata"]["temp_img1"] = cv2.bitwise_and(data["image"], data["image"], mask=white_mask)
    data["metadata"]["temp_img2"] = cv2.bitwise_and(data["image"], data["image"], mask=black_mask)

    data["image"] = cv2.bitwise_and(data["image"], data["image"], mask=complete_mask)

    #make mask 
    # bounding rect
    
    return data

