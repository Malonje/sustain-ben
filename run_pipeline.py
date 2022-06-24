import numpy as np
import torch
from sustainbench import get_dataset
import util

import predict_croptype
import predict_cropseg


PATH_TO_CAUVERY_IMAGES = None  # TODO
PATH_TO_CROP_TYPE_MAPPING_WEIGHTS = None  # TODO


def run(args):
    mapped_crops = predict_croptype.main(args)
    delineated_patches = predict_cropseg.main(args)

    paddy_fields = extract_paddy_fields(mapped_crops, delineated_patches)

    sowing_dates, transplanting_dates, harvesting_dates = crop_sowing_harvesting_date_prediction(paddy_fields)
    paddy_fields = clip_seasonal_paddy_images(paddy_fields, transplanting_dates, harvesting_dates)
    yield_estimations = crop_yield_prediction(paddy_fields)


if __name__ == "__main__":
    parser = util.get_train_parser()
    run(parser.parse_args())
