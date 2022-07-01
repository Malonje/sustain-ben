from .version import __version__
from .get_dataset import get_dataset

benchmark_datasets = [
    'poverty',
    'fmow',
    'africa_crop_type_mapping',
    'crop_seg',
    'crop_yield'
]

additional_datasets = [
    'crop_sowing_transplanting_harvesting'
]

supported_datasets = benchmark_datasets + additional_datasets
