import glob

__all__ = [
    'PATH_TO_DATA',
    'GUIS_CODE',
    'GUIS_SKETCH',
]


PATH_TO_DATA = '../data/all_data/'
GUIS_CODE = glob.glob(PATH_TO_DATA + '*.gui')
GUIS_SKETCH = glob.glob(PATH_TO_DATA + '*.png')

assert len(GUIS_CODE) == len(GUIS_SKETCH)
