import math
import cv2
import utils.utils as Utils
import numpy as np
import singleSquaresProcessing.parameters as singleParams

# cornerCutSize -> size of the corners previously cut from the image
# resultsMatrixFieldName -> name of the field in metadata where the results matrix were stored
# firstwarpMatrixFieldName -> name of the field in metadata where the first warp matrix used to obtain the near straight grid layout previously was stored
# secondWarpMatrixFieldName -> name of the field in metadata where the second warp matrix used to obtain the final straight grid layout previously was stored
# rotationMatrixFieldName -> name of the field in metadata where the rotation matrix calculated from little horse was stored
# cornersFieldName -> name of the field in metadata where the occupied squares corners will be stored
def get_occupied_squares_corners(data, resultsMatrixFieldName="chessboard_matrix", firstWarpMatrixFieldName="warp_matrix", secondWarpMatrixFieldName="refined_warp_matrix", rotationMatrixFieldName="rotation_matrix", cornersFieldName="occupied_squares_corners"):
    results_matrix = data["metadata"].get(resultsMatrixFieldName, None)
    warp_matrix_1 = data["metadata"].get(firstWarpMatrixFieldName, None)
    warp_matrix_2 = data["metadata"].get(secondWarpMatrixFieldName, None)
    rotation_matrix_2d = data["metadata"].get(rotationMatrixFieldName, None)
    
    if results_matrix is None:
        raise ValueError("Results matrix data must be defined previously in pipeline, in order to get occupied squares corners")
    if warp_matrix_1 is None or warp_matrix_2 is None:
        raise ValueError("Warp matrixes data must be defined previously in pipeline, in order to get occupied squares corners")
    if rotation_matrix_2d is None: 
        raise ValueError("Rotation matrix data must be defined previously in pipeline, in order to get occupied squares corners")

    img_h, img_w = data["image"].shape[:2]
    square_h, square_w = img_h// 8, img_w // 8

    #3x3 matrix for 2d affine
    rotation_matrix_3d = np.vstack([rotation_matrix_2d, [0,0,1]])

    inv_warp_matrix_2 = np.linalg.inv(warp_matrix_2) # used for the last un-warping step in the image
    inv_rotation_matrix = np.linalg.inv(rotation_matrix_3d) # used for un-rotating the image
    inv_warp_matrix_1 = np.linalg.inv(warp_matrix_1) # used for initial un-warping the image

    final_matrix = inv_warp_matrix_1 @ inv_rotation_matrix @ inv_warp_matrix_2 #final matrix to apply, that inverts both transformations # first undo the rotation, then undo the warping

    #calculate corners of occupied squares
    final_squares_corners = []
    for row in range(8):
        for col in range(8):
            if results_matrix[7- row][col] != 0: # we read the matrix from bottom to top, so we need to invert the row index

                #calculate the corners of the occupied square
                x1 = col * square_w
                y1 = row * square_h
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
# bbFieldName: name of the field in metadata where the static bounding boxes were stored
# rawContourFieldName: name of the field in metadata where the raw contour of the board (without convex hull and aprox poly processing, directly from findContours()) was be stored
# refinedBbFieldName: name of the field in metadata where the refined boundinb boxes will be stored
# whiteLowerBound: lower bound for the white piece color in HSV
# whiteUpperBound: upper bound for the white piece color in HSV
# blackLowerBound: lower bound for the black piece color in HSV
# blackUpperBound: upper bound for the black piece color in HSV
# whiteEdgesKernel: size of the kernel for the white piece edges erosion
# blackEdgesKernel: size of the kernel for the black piece edges erosion
# resultsMatrixFieldName: name of the field in metadata where the results matrix was stored

def refine_bounding_boxes(data, bbFieldName="bounding_boxes", rawContourFieldName="board_contour_raw", refinedBbFieldName="refined_bounding_boxes", whiteLowerBound=(20, 60, 50), whiteUpperBound=(25, 150, 255), blackLowerBound=(10, 50, 60), blackUpperBound=(20, 255, 255), whiteEdgesKernel=(5,5), blackEdgesKernel=(5,5), pieceMaskMinArea=4000, pieceMaskMaxCenterDist=0.25, resultsMatrixFieldName="chessboard_matrix"):
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
    
    raw_contour = data["metadata"].get(rawContourFieldName, None)
    if (raw_contour is None):
        raise ValueError("Raw contour data must be defined previously in pipeline, in order to find piece contours")
    
    # create mask from contour, to filter out everything that isnt part of the chessboard/pieces, out of other masks
    board_mask = _get_mask_from_board_outline(data["image"].shape[:2], raw_contour) # crop the white mask with the board outline
    
    #flip the results matrix to match the image orientation
    results_matrix = np.flipud(results_matrix)

    # remove all 0s and flatten results matrix, to get only the color of the occupied squares
    occupied_squares = results_matrix[results_matrix != 0]

    refined_bboxes = []

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox

        # get the bounding box of the piece in the original image
        piece_bbox = data["image"][y1:y2, x1:x2]

        # cv2.imshow("piece_bbox", piece_bbox)

        # get the bounding box of the piece in the board mask
        piece_mask_bbox = board_mask[y1:y2, x1:x2]
        
        piece_mask = None
        refined_bbox = None

        # if occupying piece is white, use white color mask
        if (occupied_squares[i] == 1):
            piece_mask = _get_color_mask(piece_bbox, piece_mask_bbox, lowerBound=whiteLowerBound, upperBound=whiteUpperBound, kernelFactor=whiteEdgesKernel, iterFactor=5, morphologyType=cv2.MORPH_OPEN)
            best_contour = _find_best_contour(piece_mask, min_area=pieceMaskMinArea, max_center_dist=pieceMaskMaxCenterDist)
            
            # if no piece identified, we might have misslabled the piece color, so we try the black mask
            if (best_contour is None):
                piece_mask = _get_color_mask(piece_bbox, piece_mask_bbox, lowerBound=blackLowerBound, upperBound=blackUpperBound, kernelFactor=blackEdgesKernel, iterFactor=5, morphologyType=cv2.MORPH_CLOSE)
                # refine mask to remove tile lines noise
                refined_piece_mask = _apply_morphology_ex(piece_mask)
                best_contour = _find_best_contour(refined_piece_mask, min_area=pieceMaskMinArea, max_center_dist=pieceMaskMaxCenterDist)
        
        # if occupying piece is black, use black color mask
        else:
            piece_mask = _get_color_mask(piece_bbox, piece_mask_bbox, lowerBound=blackLowerBound, upperBound=blackUpperBound, kernelFactor=blackEdgesKernel, iterFactor=5, morphologyType=cv2.MORPH_CLOSE)
            refined_piece_mask = _apply_morphology_ex(piece_mask)
            best_contour = _find_best_contour(refined_piece_mask, min_area=pieceMaskMinArea, max_center_dist=pieceMaskMaxCenterDist)
            
            # if no piece identified, we might have misslabled the piece color, so we try the white mask
            if (best_contour is None):
                piece_mask = _get_color_mask(piece_bbox, piece_mask_bbox, lowerBound=whiteLowerBound, upperBound=whiteUpperBound, kernelFactor=whiteEdgesKernel, iterFactor=5, morphologyType=cv2.MORPH_OPEN)
                best_contour = _find_best_contour(piece_mask, min_area=pieceMaskMinArea, max_center_dist=pieceMaskMaxCenterDist)
       
        
        #find best contour from image mask mask
        if (best_contour is None):
            refined_bboxes.append(bbox)
            continue
    
        x,y,w,h = cv2.boundingRect(best_contour) # get the bounding box of the biggest contour
        refined_bbox = [x + x1, y + y1, x + x1 + w, y + y1 + h] # add the offset of the original bounding box to the new bounding box
        refined_bboxes.append(refined_bbox)

    data["metadata"][refinedBbFieldName] = refined_bboxes

    return data

# apply a color mask to the image, to get the pieces
# board_mask: additional mask to only consider pixels inside the chessboard in the final mask 
# lowerBound: lower bound for the color in HSV
# upperBound: upper bound for the color in HSV
# kernelFactor: size of the kernel for the morphological operation
# kernelType: type of kernel to use for the morphological operation (cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS) - ellipse shape helps reduce line noise
# iterFactor: number of iterations for the morphological operation
# morphologyType: type of morphological operation to apply (cv2.MORPH_OPEN or cv2.MORPH_CLOSE)
def _get_color_mask(image, board_mask=None, lowerBound=(0,0,0), upperBound=(128,128,128), kernelType=cv2.MORPH_ELLIPSE, kernelFactor=(5,5), iterFactor=3, morphologyType=cv2.MORPH_OPEN):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_image, np.array(list(lowerBound)), np.array(list(upperBound)))
    _, thresh = cv2.threshold(mask, 0, 255, 0)

    # Crop the threshold mask with the board region mask
    if (board_mask is not None):
        thresh = cv2.bitwise_and(thresh, board_mask)
    
    kernel = cv2.getStructuringElement(kernelType, kernelFactor)
    thresh = cv2.morphologyEx(thresh, morphologyType, kernel, iterations=iterFactor) # we open because complete white pieces are easy to detect, so we just want to remove the noise

    return thresh

#apply morphological operation to the mask (used specifically for dark pieces mask)
def _apply_morphology_ex(mask, kernelType=cv2.MORPH_ELLIPSE, kernelSize=(5,5), iterations=5, morphologyType=cv2.MORPH_OPEN):

    kernel = cv2.getStructuringElement(kernelType, kernelSize)
    result_mask = cv2.morphologyEx(mask, morphologyType, kernel, iterations=iterations) 
    return result_mask

# find the best contour from the piece mask
# min_area: minimum area of the contour to be considered
# max_center_dist: maximum distance from the center of the image to the center of the contour (0-0.5)
def _find_best_contour(piece_mask, min_area=4000, max_center_dist=0.25):
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # cv2.imshow("piece_mask", piece_mask)
    h, w = piece_mask.shape[:2]
    center = (w // 2, int(h * 2/3)) # horizontally center, vertically on the lower third of the bbox
    best_contour = None # for debug, can be removed later
    best_area = float(0)

    for cnt in contours:
        cnt_bbox = cv2.boundingRect(cnt)
        x, y, bw, bh = cnt_bbox
        cx, cy = x + bw/2, y + bh/2
        area = cv2.contourArea(cnt)
        dist = np.linalg.norm(center - np.array([cx, cy])) / max(w, h) # calculate distance from center of image to center of contour

        if dist < max_center_dist and area > min_area and area > best_area:
            best_area = area
            best_contour = cnt

    if best_contour is None:
        return None

    return best_contour

def _get_mask_from_board_outline(shape, board_outline):

    # create mask from contour, to filter out everything that isnt part of the chessboard/pieces, out of other masks
    board_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(board_mask, board_outline, -1, (255, 255, 255), -1)

    return board_mask
