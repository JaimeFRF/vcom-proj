from functools import partial
from utils.metadataMerger import MetadataMerger
from utils.utils import *
from IO.json_handler import *

from preProcessing.functions import *
import preProcessing.parameters as preProcParams
from preProcessing.preProcessing import PreProcessing
from squaresProcessing.functions import *
import squaresProcessing.parameters as squaresParams
from squaresProcessing.squaresProcessing import SquaresProcessing
from rotationProcessing.functions import *
import rotationProcessing.parameters as rotParams
from rotationProcessing.rotationProcessing import RotationProcessing
from singleSquaresProcessing.functions import *
import singleSquaresProcessing.parameters as singleParams
from singleSquaresProcessing.singleSquaresProcessing import SingleSquaresProcessing


## Pipeline Design Pattern -> Só é preciso meter as funções/ordem etc que queremos
# NOTE: if you want to specify certain attributes in pipeline do partial(func_name, arg1=value1, arg2=value2,...)

pp_pipeline = PreProcessing([
    # partial(show_current_image, imageTitle="Original image", resizeAmount=0.25),
    convert_to_gray,
    partial(gaussian, ksize=preProcParams.gaussian_ksize), # isto devia ser bilateral filter maybe?
    # incluir aqui tmb o equalizeHist ?
])

#identify the board, warp it to occupy the whole image
squares_pipeline = SquaresProcessing([
    partial(canny, low=squaresParams.canny_low_threshold, high=squaresParams.canny_high_threshold),
    # partial(show_current_image, imageTitle="Canny Edges", resizeAmount=0.25),
    partial(dilate_edges, ksize=squaresParams.dilate_ksize, iterations=squaresParams.dilate_iterations),
    # partial(show_current_image, imageTitle="Canny Dilated", resizeAmount=0.25),
    partial(closing, ksize=squaresParams.closing_ksize, iterations=squaresParams.closing_iterations),
    # partial(show_current_image, imageTitle="Canny Dilated Closed", resizeAmount=0.25),
    partial(find_board_countour_and_corners, approxPolyDP_epsilon=squaresParams.approxPolyDP_epsilon),
    # partial(draw_contours, imageTitle="Original with Countors", fieldName="board_contour"),
    partial(warp_image_from_board_corners,warp_width=squaresParams.warp_width, warp_height=squaresParams.warp_height),
    # partial(hough_lines, rho=squaresParams.hough_rho, theta=squaresParams.hough_theta, votes=squaresParams.hough_votes),
    # partial(draw_hough_lines, color=Utils.color_red, withText=False)
    # partial(show_current_image, imageTitle="Warped final pipeline image", resizeAmount=0.25),
    partial(save_current_image_in_metadata, fieldName="warped_image"), # save image to metadata to be reused later
])

#separate small pipeline for the horse reference image, and the results will be merged with the main pipeline
separate_horse_pipeline = RotationProcessing([
    convert_to_gray,
    equalizeHist,
    partial(save_image_dimensions_in_metadata, widthFieldTitle="horse_width", heightFieldTitle="horse_height"), # save this data to be used to calculate the homography in the rotate_pipeline
    partial(sift, keypointsFieldTitle="horse_keypoints", descriptorsFieldTitle="horse_descriptors"), # store calculated keypoints and descriptors in metadata fields, accessible by other functions in the main pipeline
    # partial(show_current_image, imageTitle="Query Image")
])

#rotate the board to the correct orientation
rotate_pipeline = RotationProcessing([
    # partial(show_current_image, imageTitle="Warped Image"),
    convert_to_gray,
    equalizeHist,
    sift,
    partial(flann_matcher, descriptors1="horse_descriptors"),
    partial(find_homography_from_matches, keypoints1="horse_keypoints"),
    # partial(draw_perspective_transformed_points, widthTitle="horse_width", heightTitle="horse_height"),
    # draw_crosshair,
    partial(set_current_image, imageFieldName="warped_image"), # set the current image to the previous warped image, to recover colors, before rotation
    rotate_img_from_homography,
])

#extract the single squares from the board, and classify them
single_squares_pipeline = SingleSquaresProcessing([
    cut_corners,
    draw_grid, 
    separate_squares,
    show_all_separate_squares,
    partial(show_separate_square, index=62),
    calculate_matrix_representation,
    partial(print_field_value, fieldName="chessboard_matrix"),
    partial(print_field_value, fieldName="total_black", withFieldName=True),
    partial(print_field_value, fieldName="total_white", withFieldName=True)
])

pre_proc_imgs = pp_pipeline.apply(read_images())
squares_results = squares_pipeline.apply(pre_proc_imgs)

separate_horse_results = separate_horse_pipeline.apply(read_single_image("our_images/cavalinhoPequeno.jpg"))[0]

squares_and_horse_results = MetadataMerger.merge_pipelines_metadata(squares_results, separate_horse_results)
rotate_results = rotate_pipeline.apply(squares_and_horse_results)
single_square_results = single_squares_pipeline.apply(rotate_results)
show_images(single_square_results)