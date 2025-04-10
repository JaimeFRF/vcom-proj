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

# def draw_piece_bounding_boxes_from_matrix(data, matrixFieldName="chessboard_matrix"):
#     image = data['image']
#     matrix = data["metadata"].get(matrixFieldName, None)  

#     matrix = np.flipud(matrix) 

#     output = image.copy()
#     h, w = output.shape[:2]
#     square_h = h // 8
#     square_w = w // 8

#     for row in range(8):
#         for col in range(8):
#             val = matrix[row][col]
#             if val != 0:
#                 x1 = col * square_w
#                 y1 = row * square_h
#                 x2 = x1 + square_w
#                 y2 = y1 + square_h

#                 color = (255, 255, 255) if val == 1 else (0, 0, 0)
#                 label = 'W' if val == 1 else 'B'

#                 cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(output, label, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     data['image'] = output
#     return data