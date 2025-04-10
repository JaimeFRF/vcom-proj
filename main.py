from functools import partial
from utils.metadataMerger import MetadataMerger
from utils.utils import *
from IO.json_handler import *
from tests.test_implementation import *

from preProcessing.functions import *
import preProcessing.parameters as preProcParams
from preProcessing.preProcessing import PreProcessing
from boardOutlineProcessing.functions import *
import boardOutlineProcessing.parameters as boardOutParams
from boardOutlineProcessing.boardOutlineProcessing import BoardOutlineProcessing
from rotationProcessing.functions import *
import rotationProcessing.parameters as rotParams
from rotationProcessing.rotationProcessing import RotationProcessing
from singleSquaresProcessing.functions import *
import singleSquaresProcessing.parameters as singleParams
from singleSquaresProcessing.singleSquaresProcessing import SingleSquaresProcessing


## Pipeline Design Pattern -> Só é preciso meter as funções/ordem etc que queremos
# NOTE: if you want to specify certain attributes in pipeline do partial(func_name, arg1=value1, arg2=value2,...)
# NOTE: the pipelines is split into 4 classes, to make it easier to understand, although they all do the same thing. The 4 classes refer to 4 different steps: 
# - preprocessing,
# - warping the image into a straight grid
# - rotating the image to face the correct way
# - split the tiles and classify them 
# NOTE: the functions and parameters for each step are organized by folder. The paramteres files are to define the default values for each function, so we can easily change them in one place (inside each step folder)
# NOTE: included a lot of the debug functions that were already in the jupyter notebook, but are commented out
# NOTE: sometimes we need to resort back to previous values or images, so i use metadata for that, to pass data between functions in pipeline
# NOTE: also, sometimes we have to proecss images separately from the main pipeline, like the little horse image, so we have a separate pipeline for that, and then merge the metadata with the main pipeline

pp_pipeline = PreProcessing([
    partial(color_mask,lower_color_bound=preProcParams.lower_color_bound, upper_color_bound=preProcParams.upper_color_bound, mask_edges_erosion=preProcParams.mask_edges_erosion),
    convert_to_gray,
    partial(bilateral, ksize=preProcParams.bilateral_ksize, sigmaColor=preProcParams.bilateral_sigmaColor, sigmaSpace=preProcParams.bilateral_sigmaSpace), # reduce noise, keeping edges sharp
])

#identify the board, warp it to occupy the whole image
board_outline_pipeline = BoardOutlineProcessing([
    partial(helper),
    partial(canny, low=boardOutParams.canny_low_threshold, high=boardOutParams.canny_high_threshold), 
    partial(show_current_image, imageTitle="Canny Edges", resizeAmount=0.25),
    apply_mask,
    partial(show_current_image, imageTitle="Masked", resizeAmount=0.25),
    partial(dilate_edges, ksize=boardOutParams.dilate_ksize, iterations=boardOutParams.dilate_iterations),
    partial(show_current_image, imageTitle="Canny Dilated", resizeAmount=0.25),
    partial(closing, ksize=boardOutParams.closing_ksize, iterations=boardOutParams.closing_iterations),
    partial(show_current_image, imageTitle="Canny Dilated Closed", resizeAmount=0.25),
    partial(find_board_countour_and_corners, approxPolyDP_epsilon=boardOutParams.approxPolyDP_epsilon),
    # partial(find_board_countour_and_corners_2, approxPolyDP_epsilon=boardOutParams.approxPolyDP_epsilon),

    partial(draw_contours, imageTitle="Original with Countors"),
    partial(warp_image_from_board_corners, warp_width=boardOutParams.warp_width, warp_height=boardOutParams.warp_height),
    # partial(hough_lines, rho=boardOutParams.hough_rho, theta=boardOutParams.hough_theta, votes=boardOutParams.hough_votes),
    # partial(draw_hough_lines, color=Utils.color_red, withText=False)

    

    partial(show_current_image, imageTitle="Warped final pipeline image"),
    
    #convert_to_gray,
    partial(helper),
    partial(cut_corners_all),
    partial(show_current_image, imageTitle="Warped final pipeline gey"),
    partial(binarize),
    partial(gaussian, ksize=(3, 3)),

    partial(show_current_image),
    partial(canny, low=boardOutParams.canny_low_threshold, high=boardOutParams.canny_high_threshold),
 
    #partial(dilate_edges, ksize=boardOutParams.dilate_ksize, iterations=1),

    partial(show_current_image, imageTitle="Warped final pipeline gey canny dilated"),

    partial(show_current_image, imageTitle="Warped final pipeline gey canny dilated closed"),
    #partial(find_board_countour_and_corners, approxPolyDP_epsilon=0.15, max_perim=20000),
    #partial(draw_contours_warped, imageTitle="Original with Countors"),


    partial(find_board_contour_and_corners_warp ,min_perim=10, max_perim=1000, approxPolyDP_epsilon=0.05),
    partial(draw_contours_warped, imageTitle="Detected Chessboard", contoursFieldName="board_contour"),
    

    partial(show_current_image, imageTitle="Warped final pipeline gey canny"),

    partial(getimg),
    partial(save_current_image_in_metadata, fieldName="warped_image"), # save image to metadata to be reused later
])

#separate small pipeline for the horse reference image, and the results will be merged with the main pipeline, in the rotation pipeline part
separate_horse_pipeline = RotationProcessing([
    convert_to_gray,
    partial(gaussian, ksize=(3, 3)),
    #equalizeHist,
    clahe,
    partial(save_image_dimensions_in_metadata, widthFieldTitle="horse_width", heightFieldTitle="horse_height"), # save this data to be used to calculate the homography in the rotate_pipeline
    partial(sift, keypointsFieldTitle="horse_keypoints", descriptorsFieldTitle="horse_descriptors"), # store calculated keypoints and descriptors in metadata fields, accessible by other functions in the main pipeline
    # partial(draw_points, imageTitle="Horse Keypoints", pointsFieldName="horse_keypoints"),
    # partial(show_current_image, imageTitle="Query Image")
])

#rotate the board to the correct orientation
rotate_pipeline = RotationProcessing([
    # partial(show_current_image, imageTitle="Warped Image"),
    convert_to_gray,
    partial(gaussian, ksize=(3,3)),
    #equalizeHist,
    clahe,
    sift,
    # partial(draw_points, imageTitle="Keypoints_main"),
    partial(flann_matcher, descriptors1="horse_descriptors"),
    partial(find_homography_from_matches, keypoints1="horse_keypoints"),
    partial(draw_perspective_transformed_points, widthTitle="horse_width", heightTitle="horse_height"),
    # draw_crosshair,
    partial(set_current_image, imageFieldName="warped_image"), # set the current image to the previous warped image, to recover colors, before rotation
    rotate_img_from_homography,
])

#extract the single squares from the board, and classify them
single_squares_pipeline = SingleSquaresProcessing([
    cut_corners,
    separate_squares,
    calculate_matrix_representation,
    partial(print_field_value, fieldName="chessboard_matrix"),
    partial(print_field_value, fieldName="total_black", withFieldName=True),
    partial(print_field_value, fieldName="total_white", withFieldName=True)
])

pre_proc_imgs = pp_pipeline.apply(read_images())
squares_results = board_outline_pipeline.apply(pre_proc_imgs)


# separate processing pipeline for the single horse image used for rotation. The metadata created here will be merged with the main pipeline results, so we can acess keypoints and descriptors of the horse in main pipeline
separate_horse_results = separate_horse_pipeline.apply(read_single_image("our_images/cavalinhoPequeno.jpg"))[0]

squares_and_horse_results = MetadataMerger.merge_pipelines_metadata(squares_results, separate_horse_results)
rotate_results = rotate_pipeline.apply(squares_and_horse_results)
single_square_results = single_squares_pipeline.apply(rotate_results)

show_debug_images(single_square_results, gridFormat=True, gridImgSize=5, gridSaveFig=False)
# show_images(squares_results)
# test_implementation(single_square_results)