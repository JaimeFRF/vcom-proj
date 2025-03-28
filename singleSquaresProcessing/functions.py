import cv2
import utils.utils as Utils
import math
import singleSquaresProcessing.parameters as Parameters
import numpy as np
import os

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
    for j in range(8):
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
        cv2.imshow(f"Square {i//8}-{i%8}", square)
    
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
    cv2.imshow(f"Square {index//8}-{index%8}", square)

    return data

# check for black pieces in white squares
def check_B_in_W(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image (black = 0, white = 255)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Count black and white pixels
    black_pixels = np.sum(thresh == 0)  # Pixels with value 0
    white_pixels = np.sum(thresh == 255)  # Pixels with value 255
    
    # Return True if black pixels exceed white pixels
    return black_pixels > 650



# check for white pieces in white squares
def check_W_in_W(img):

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV", hsv)

    # Create a mask:
    # - S (saturation) < 82 (low saturation for white/light colors)
    # - V (value/brightness) < 101 (darker areas excluded)
    # Pixels within this range will be 255 (white), others 0 (black)
    upper_bound = (179, 255, 255)    # Minimum H, S, V
    lower_bound = (0, 82, 101)  # Maximum H, S, V
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #print_resized_up("Mask", mask)

    # Count white pixels (255) and black pixels (0) in the mask
    white_pixels = np.sum(mask == 255)
    black_pixels = np.sum(mask == 0)

    # Return True if more white pixels than black (indicating a light area/piece)
    return white_pixels > 650

# check for white pieces in black squares
def check_W_in_B(img):

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)

    # Create a mask:
    # - S (saturation) < 82 (low saturation for white/light colors)
    # - V (value/brightness) < 101 (darker areas excluded)
    # Pixels within this range will be 255 (white), others 0 (black)
    upper_bound = (179, 255, 255)    # Minimum H, S, V
    lower_bound = (0, 49, 80)  # Maximum H, S, V
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Count white pixels (255) and black pixels (0) in the mask
    white_pixels = np.sum(mask == 255)
    black_pixels = np.sum(mask == 0)
    # print(f"White pixels: {white_pixels}, Black pixels: {black_pixels}")

    # Return True if more white pixels than black (indicating a light area/piece)
    return white_pixels > 650

# check for black pieces in black squares
def check_B_in_B(img):

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV", hsv)

    # Create a mask:
    # - S (saturation) > 82 (low saturation for white/light colors)
    # - V (value/brightness) < 101 (darker areas excluded)
    # Pixels within this range will be 255 (white), others 0 (black)
    upper_bound = (179, 255, 30)    # Minimum H, S, V
    lower_bound = (0, 0, 0)  # Maximum H, S, V
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #print_resized_up("Mask", mask)

    # Count white pixels (255) and black pixels (0) in the mask
    white_pixels = np.sum(mask == 255)
    black_pixels = np.sum(mask == 0)
    #print(f"White pixels: {white_pixels}, Black pixels: {black_pixels}")

    # Return True if more white pixels than black (indicating a light area/piece)
    return white_pixels > 650

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
            row = i // 8
            col = i % 8
            # Check if the tile is black or white based on its position
            if i % 2 == 0:  # Black tile
                if check_B_in_B(tile):
                    # print(f"Tile {row}_{col} is a black piece.")
                    blacks += 1
                    chessboard_matrix[row, col] = 2
                elif check_W_in_B(tile):
                    # print(f"Tile {row}_{col} is a white piece.")
                    whites += 1
                    chessboard_matrix[row, col] = 1
            else:  # White tile
                if check_W_in_W(tile):
                    # print(f"Tile {row}_{col} is a white piece.")
                    whites += 1
                    chessboard_matrix[row, col] = 1
                elif check_B_in_W(tile):
                    # print(f"Tile {row}_{col} is a black piece.")
                    blacks += 1
                    chessboard_matrix[row, col] = 2

    # save results
    data["metadata"][totalBlackFieldName] = blacks
    data["metadata"][totalWhiteFieldName] = whites
    data["metadata"][matrixFieldName] = chessboard_matrix

    print("Chessboard Matrix (1 for white piece, 2 for black piece, 0 for empty)")

    return data


