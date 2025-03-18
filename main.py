from utils.utils import *
from functools import partial
from IO.json_handler import read_images
from preProcessing.functions import *
import preProcessing.parameters as preProcParams
from squaresProcessing.functions import *
import squaresProcessing.parameters as squaresParams
from preProcessing.preProcessing import PreProcessing
from squaresProcessing.squaresProcessing import SquaresProcessing

## Pipeline Design Pattern -> Só é preciso meter as funções/ordem etc que queremos
# NOTE: if you want to specify certain attributes in pipeline do partial(func_name, arg1=value1, arg2=value2,...)

pp_pipeline = PreProcessing([
    convert_to_gray,
    partial(bilateral,d=preProcParams.bilateral_d, sigmaColor=preProcParams.bilateral_sigmaColor, sigmaSpace=preProcParams.bilateral_sigmaSpace),
    normalize,
    equalizeHist
])

squares_pipeline = SquaresProcessing([
    partial(canny, low=squaresParams.canny_low_threshold, high=squaresParams.canny_high_threshold),
    partial(hough_lines, rho=squaresParams.hough_rho, theta=squaresParams.hough_theta, votes=squaresParams.hough_votes),
    partial(draw_hough_lines, color=Utils.color_red, withText=False)
])

imgs = read_images()
pre_proc_imgs = pp_pipeline.apply(imgs)
squares_results = squares_pipeline.apply(pre_proc_imgs)
show_images(squares_results)