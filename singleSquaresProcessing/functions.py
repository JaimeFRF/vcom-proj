import cv2
import utils.utils as Utils
import math
import singleSquaresProcessing.parameters as Parameters
import numpy as np
import os

## Module for functions related to splitting the board into tiles, processing the tiles, and classifying the pieces in them

#cut image corners all around, by certain amount
def cut_corners(data, cut_size=40):
    data["image"] = data["image"][cut_size:460, cut_size:460]
    return data

#separate individual squares from the board
#assumes current image is already warped into correct grid, and ready to be cut
def separate_squares(data, squareFieldName="squares_list", save_tiles=False):

    #TODO: não se podia dividir só por 8 com o tamanho atual? size = round(data["image"].shape[0]) / 8
    size = round((465-44) / 8) 

    img = data["image"]
    squares_list = []

    if (save_tiles):
        os.makedirs(Parameters.tiles_save_dir, exist_ok=True)

    # extract each tile
    for j in range(7, -1 , -1):
        for i in range(8):
            
            # Extract the tile
            x1, y1 = i * size, j * size
            x2, y2 = x1 + size, y1 + size
            
            # get the tile image
            square = img[y1:y2, x1:x2].copy()

            # save the tile image
            if (save_tiles):
                save_dir = f"{Parameters.tiles_save_dir}/tile_{i}_{j}.png"
                cv2.imwrite(save_dir, square)

            squares_list.append(square)
        
    data["metadata"][squareFieldName] = squares_list
    
    return data

def show_all_separate_squares(data, squaresListFieldName="squares_list"):
    squares_list = data["metadata"].get(squaresListFieldName, None)
    if (squares_list is None):
        raise ValueError("Squares list must be defined previously in pipeline, to show all squares")
    
    for i, square in enumerate(squares_list):
        Utils.show_image_with_name(data, f"Square {i//8}-{i%8}", square)
    
    return data

# show cut square with specific index, from the squares list, starting at 0 in top left, 
# index grows to the right then down
def show_separate_square(data, index, squaresListFieldName="squares_list"):
    squares_list = data["metadata"].get(squaresListFieldName, None)
    if (squares_list is None):
        raise ValueError("Squares list must be defined previously in pipeline, to show all squares")
    
    if index < 0 or index >= len(squares_list):
        raise ValueError(f"Index {index} out of bounds for squares list of length {len(squares_list)}")
    
    square = squares_list[index]
    Utils.show_image_with_name(data, f"Square {index//8}-{index%8}", square)

    return data

# from split tiles of board, identify occupied tiles and which color piece it contains
# squaresListFieldName: name of the field in metadata where the list of squares is stored
# matrixFieldName: name of the field in metadata where the matrix representation will be stored 
#                   0 - empty, 1 - white piece, 2 - black piece
# totalBlackFieldName: name of the field in metadata where the total number of black pieces will be stored
# totalWhiteFieldName: name of the field in metadata where the total number of white pieces will be stored
def calculate_matrix_representation(data, squaresListName="squares_list", matrixFieldName="chessboard_matrix", totalBlackFieldName="total_black", totalWhiteFieldName="total_white"):
    squares_list = data["metadata"].get(squaresListName, None)
    if (squares_list is None):
        raise ValueError("Squares list must be defined previously in pipeline, to show all squares")
    
    # Initialize counters for black and white pieces
    blacks = 0
    whites = 0

    # Initialize the 8x8 matrix (0 for empty, 1 for piece)
    chessboard_matrix = np.zeros((8, 8), dtype=int)

    for i, tile in enumerate(squares_list):
            grey = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            # TODO: Tunar esta situação
            circle = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1,20, param1=50,param2=25,minRadius=0,maxRadius=0)
            row = i // 8
            col = i % 8
            
            if circle is not None and circle.size > 0:
                circle = np.uint16(np.around(circle))

                for i in circle[0,:]:
                    width, height, _ = tile.shape
                    (ix, iy) = width // 2, height // 2
                    cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                    
                    if(abs(cx - ix) <= 10 and abs(cy - iy) <= 10):
                        crop = grey[(cy-r):(cy+r), (cx - r):(cx + r)]

                        mean_intensity = np.mean(crop)
                        if mean_intensity > 100:
                            whites += 1
                            chessboard_matrix[row, col] = 1
                        else:
                            blacks += 1
                            chessboard_matrix[row, col] = 2

    # save results
    data["metadata"][totalBlackFieldName] = blacks
    data["metadata"][totalWhiteFieldName] = whites
    data["metadata"][matrixFieldName] = chessboard_matrix

    print("Chessboard Matrix (1 for white piece, 2 for black piece, 0 for empty)")

    return data


