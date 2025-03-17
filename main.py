from IO.json_handler import read_images
from utils.utils import *
from preProcessing.functions import *
from functools import partial
from preProcessing.preProcessing import PreProcessing


## Pipeline Design Pattern -> Só é preciso meter as funções/ordem etc que queremos
pp_pipeline = PreProcessing([
    partial(gaussian, ksize=(25, 25)), #if you want to specify certain attributes
    convert_to_gray,
    normalize,
    equalizeHist
])

show_images(pp_pipeline.apply(read_images()))