import math
import cv2
import utils.utils as Utils
import numpy as np


def draw_piece_bounding_boxes_from_matrix(data, matrixFieldName="chessboard_matrix"):
    image = data['image']
    matrix = data["metadata"].get(matrixFieldName, None)  

    matrix = np.flipud(matrix) 

    output = image.copy()
    h, w = output.shape[:2]
    square_h = h // 8
    square_w = w // 8

    for row in range(8):
        for col in range(8):
            val = matrix[row][col]
            if val != 0:
                x1 = col * square_w
                y1 = row * square_h
                x2 = x1 + square_w
                y2 = y1 + square_h

                color = (255, 255, 255) if val == 1 else (0, 0, 0)
                label = 'W' if val == 1 else 'B'

                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, label, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    data['image'] = output
    return data