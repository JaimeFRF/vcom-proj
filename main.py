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

## Pipeline Design Pattern

pp_pipeline = PreProcessing([
    partial(save_current_image_in_metadata, fieldName="og_img"),
    partial(color_mask,lower_color_bound=preProcParams.lower_color_bound, upper_color_bound=preProcParams.upper_color_bound, mask_edges_erosion=preProcParams.mask_edges_erosion),
    convert_to_gray,
    partial(bilateral, ksize=preProcParams.bilateral_ksize, sigmaColor=preProcParams.bilateral_sigmaColor, sigmaSpace=preProcParams.bilateral_sigmaSpace), # reduce noise, keeping edges sharp
])

#identify the board, warp it to occupy the whole image
board_outline_pipeline = BoardOutlineProcessing([
    partial(canny, low=boardOutParams.canny_low_threshold, high=boardOutParams.canny_high_threshold), 
    apply_mask,
    partial(dilate_edges, ksize=boardOutParams.dilate_ksize, iterations=boardOutParams.dilate_iterations),
    partial(save_current_image_in_metadata, fieldName="dilated_image"),
    partial(closing, ksize=boardOutParams.closing_ksize, iterations=boardOutParams.closing_iterations),
    partial(find_board_countour_and_corners, approxPolyDP_epsilon=boardOutParams.approxPolyDP_epsilon),
    partial(warp_image_from_board_corners, warp_width=boardOutParams.warp_width, warp_height=boardOutParams.warp_height),
    partial(save_current_image_in_metadata, fieldName="warped_image"), # save image to metadata to be reused later
])

#separate small pipeline for the horse reference image, and the results will be merged with the main pipeline, in the rotation pipeline part
separate_horse_pipeline = RotationProcessing([
    convert_to_gray,
    partial(gaussian, ksize=(3, 3)),
    clahe,
    partial(save_image_dimensions_in_metadata, widthFieldTitle="horse_width", heightFieldTitle="horse_height"), # save this data to be used to calculate the homography in the rotate_pipeline
    partial(sift, keypointsFieldTitle="horse_keypoints", descriptorsFieldTitle="horse_descriptors"), # store calculated keypoints and descriptors in metadata fields, accessible by other functions in the main pipeline
])

#rotate the board to the correct orientation
rotate_pipeline = RotationProcessing([
    convert_to_gray,
    partial(gaussian, ksize=(3,3)),
    clahe,
    sift,
    partial(flann_matcher, descriptors1="horse_descriptors"),
    partial(find_homography_from_matches, keypoints1="horse_keypoints"),
    partial(set_current_image, imageFieldName="warped_image"), # set the current image to the previous warped image, to recover colors, before rotation
    rotate_img_from_homography,
    partial(save_current_image_in_metadata, fieldName="warped_rotated_image")
])

# detect grid region, based on lines density
density_pipeline = DensityProcessing([
    convert_to_gray,
    partial(gaussian, ksize=(5,5)),
    partial(canny, low=100, high=200), 
    hough_lines,
    draw_hough_lines_on_warped_image,
    calculate_corners,
    partial(set_current_image, imageFieldName="warped_rotated_image"), 
    partial(warp_image_from_board_corners, imgFieldName = "image", warp_width=boardOutParams.warp_width, warp_height=boardOutParams.warp_height, warpMatrixFieldName="refined_warp_matrix"),
    partial(save_current_image_in_metadata, fieldName="final_grid"),
    convert_to_gray,
    separate_squares,
])

# identify individual occupied tiles
single_squares_pipeline = SingleSquaresProcessing([
    calculate_matrix_representation
])

# calculate piece bounding boxes
draw_boxes_pipeline = BoundingBoxes([
    partial(set_current_image, imageFieldName="final_grid"),
    get_occupied_squares_corners,
    partial(get_pieces_static_bounding_boxes, bbVertFactor=boundingParams.bbVertFactor),
    partial(set_current_image, imageFieldName="og_img"),
    partial(bilateral, ksize=boundingParams.bilateral_ksize, sigmaColor=boundingParams.bilateral_sigmaColor), # reduce noise, keeping edges sharp
    partial(gamma_adjust, gamma=boundingParams.gamma, cutoff=boundingParams.cutoff),
    partial(refine_bounding_boxes, whiteLowerBound=boundingParams.white_lower_bound, whiteUpperBound=boundingParams.white_upper_bound, blackLowerBound=boundingParams.black_lower_bound, blackUpperBound=boundingParams.black_upper_bound, whiteEdgesKernel=boundingParams.white_edges_kernel, blackEdgesKernel=boundingParams.black_edges_kernel, pieceMaskMinArea=boundingParams.piece_min_area, pieceMaskMaxCenterDist=boundingParams.piece_max_center_dist)
])

#running pipelines
pre_proc_imgs = pp_pipeline.apply(read_images())
squares_results = board_outline_pipeline.apply(pre_proc_imgs)
separate_horse_results = separate_horse_pipeline.apply(read_single_image("our_images/cavalinhoPequeno.jpg"))[0] # separate processing pipeline for the single horse image used for rotation. The metadata created here will be merged with the main pipeline results, so we can acess keypoints and descriptors of the horse in main pipeline
squares_and_horse_results = MetadataMerger.merge_pipelines_metadata(squares_results, separate_horse_results)
rotate_results = rotate_pipeline.apply(squares_and_horse_results)
density_results = density_pipeline.apply(rotate_results)
single_square_results = single_squares_pipeline.apply(rotate_results)
final_results = draw_boxes_pipeline.apply(single_square_results)

# test_implementation(final_results) # tests
write_results(single_square_results)

