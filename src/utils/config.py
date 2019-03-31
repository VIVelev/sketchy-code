import glob

__all__ = [
    'PATH_TO_DATA',
    'GUIS_CODE',
    'GUIS_SKETCH',
    'IMAGE_SIZE',
]


PATH_TO_DATA = '/Users/victor/Desktop/sketchy-code/data/all_data/'
GUIS_CODE = glob.glob(PATH_TO_DATA + '*.gui')
GUIS_SKETCH = glob.glob(PATH_TO_DATA + '*.png')
IMAGE_SIZE = (256, 256, 3)

assert len(GUIS_CODE) == len(GUIS_SKETCH)
