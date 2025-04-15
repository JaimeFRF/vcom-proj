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
from rotationProcessing.rotationProcessing import RotationProcessing
from singleSquaresProcessing.functions import *
from singleSquaresProcessing.singleSquaresProcessing import SingleSquaresProcessing
from boundingBoxesNEW.functions import *
import boundingBoxesNEW.parameters as boundingParams
from boundingBoxesNEW.boundingBoxesNew import BoundingBoxes
from densityProcessing.densityProcessing import *
from densityProcessing.functions import *

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
    partial(save_current_image_in_metadata, fieldName="og_img"),
    # table_segmentation,
    # partial(show_current_image, imageTitle="Prepocessed", resizeAmount=0.25),
    partial(color_mask,lower_color_bound=preProcParams.lower_color_bound, upper_color_bound=preProcParams.upper_color_bound, mask_edges_erosion=preProcParams.mask_edges_erosion),
    convert_to_gray,
    partial(bilateral, ksize=preProcParams.bilateral_ksize, sigmaColor=preProcParams.bilateral_sigmaColor, sigmaSpace=preProcParams.bilateral_sigmaSpace), # reduce noise, keeping edges sharp
])

#identify the board, warp it to occupy the whole image
board_outline_pipeline = BoardOutlineProcessing([
    partial(canny, low=boardOutParams.canny_low_threshold, high=boardOutParams.canny_high_threshold), 
    # partial(show_current_image, imageTitle="Canny Edges", resizeAmount=0.25),
    apply_mask,
    # partial(show_current_image, imageTitle="Masked", resizeAmount=0.25),
    partial(dilate_edges, ksize=boardOutParams.dilate_ksize, iterations=boardOutParams.dilate_iterations),
    partial(save_current_image_in_metadata, fieldName="dilated_image"),
    # partial(show_current_image, imageTitle="Canny Dilated", resizeAmount=0.25),
    partial(closing, ksize=boardOutParams.closing_ksize, iterations=boardOutParams.closing_iterations),
    # partial(show_current_image, imageTitle="Canny Dilated Closed", resizeAmount=0.25),
    partial(find_board_countour_and_corners, approxPolyDP_epsilon=boardOutParams.approxPolyDP_epsilon),
    # partial(draw_contours, imageTitle="Original with Countors"),
    partial(warp_image_from_board_corners, warp_width=boardOutParams.warp_width, warp_height=boardOutParams.warp_height),
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
    # partial(draw_perspective_transformed_points, widthTitle="horse_width", heightTitle="horse_height"),
    # draw_crosshair,
    partial(set_current_image, imageFieldName="warped_image"), # set the current image to the previous warped image, to recover colors, before rotation
    rotate_img_from_homography,
    partial(save_current_image_in_metadata, fieldName="warped_rotated_image")
])

density_pipeline = DensityProcessing([
    # partial(show_current_image, imageTitle="Image before density"),
    convert_to_gray,
    partial(gaussian, ksize=(5,5)),
    partial(canny, low=100, high=200), 
    hough_lines,
    draw_hough_lines_on_warped_image,
    calculate_corners,
    partial(set_current_image, imageFieldName="warped_rotated_image"), 
    # partial(show_metadata_image, imageTitle="Hough lines on warped image", imageName="line_map"),
    # partial(draw_contours, imageTitle="Warped with contours from density", imgName="image", contoursFieldName="board_contour_warped"),
    partial(warp_image_from_board_corners, imgFieldName = "image", warp_width=boardOutParams.warp_width, warp_height=boardOutParams.warp_height, warpMatrixFieldName="refined_warp_matrix"),
    # partial(show_current_image, imageTitle="Warped final pipeline image"),
    partial(save_current_image_in_metadata, fieldName="final_grid"),
    convert_to_gray,
    separate_squares,
])

single_squares_pipeline = SingleSquaresProcessing([
    calculate_matrix_representation
])

draw_boxes_pipeline = BoundingBoxes([
    partial(set_current_image, imageFieldName="final_grid"),
    get_occupied_squares_corners,
    partial(draw_cross_from_array, imageTitle="Occupied Squares Corners", pointsFieldName="debug_corners_warped_img", radius=3, thickness=3, color=Utils.color_blue),#, makeColored=True),
    partial(get_pieces_static_bounding_boxes, bbVertFactor=boundingParams.bbVertFactor),
    partial(set_current_image, imageFieldName="og_img"),
    partial(bilateral, ksize=boundingParams.bilateral_ksize, sigmaColor=boundingParams.bilateral_sigmaColor), # reduce noise, keeping edges sharp
    partial(show_current_image, imageTitle="bilateral_og_img", resizeAmount=0.25),
    partial(gamma_adjust, gamma=boundingParams.gamma, cutoff=boundingParams.cutoff),
    partial(show_current_image, imageTitle="gamma", resizeAmount=0.25),
    # partial(download_current_image, path="temp/gamma_038_256_adjust"),
    partial(draw_contours, imageTitle="Board contour mask", contoursFieldName="board_contour", color=Utils.color_red, thickness=5),
    partial(draw_contours, imageTitle="Board contour raw mask", contoursFieldName="board_contour_raw", color=Utils.color_red, thickness=5),

    partial(refine_bounding_boxes, whiteLowerBound=boundingParams.white_lower_bound, whiteUpperBound=boundingParams.white_upper_bound, blackLowerBound=boundingParams.black_lower_bound, blackUpperBound=boundingParams.black_upper_bound, whiteEdgesKernel=boundingParams.white_edges_kernel, blackEdgesKernel=boundingParams.black_edges_kernel, pieceMaskMinArea=boundingParams.piece_min_area, pieceMaskMaxCenterDist=boundingParams.piece_max_center_dist),
    partial(draw_points_from_array, imageTitle="Occupied Squares Corners", pointsFieldName="occupied_squares_corners", radius=10, thickness=10, color=Utils.color_red),#, makeColored=True),
    partial(draw_rectangles, imageTitle="Occupied Squares Bounding Boxes", fieldName="refined_bounding_boxes", color=Utils.color_green, thickness=2)#, makeColored=True),
])

pre_proc_imgs = pp_pipeline.apply(read_images())
squares_results = board_outline_pipeline.apply(pre_proc_imgs)

#show_images(squares_results)

# separate processing pipeline for the single horse image used for rotation. The metadata created here will be merged with the main pipeline results, so we can acess keypoints and descriptors of the horse in main pipeline
separate_horse_results = separate_horse_pipeline.apply(read_single_image("our_images/cavalinhoPequeno.jpg"))[0]
squares_and_horse_results = MetadataMerger.merge_pipelines_metadata(squares_results, separate_horse_results)
rotate_results = rotate_pipeline.apply(squares_and_horse_results)
density_results = density_pipeline.apply(rotate_results)
single_square_results = single_squares_pipeline.apply(rotate_results)
final_results = draw_boxes_pipeline.apply(single_square_results)

test_implementation(final_results)
show_debug_images(squares_results, gridFormat=True, gridImgSize=5, gridSaveFig=False)
# show_images(final_results)
write_results(single_square_results)

