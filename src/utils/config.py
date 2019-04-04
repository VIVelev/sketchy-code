import glob

__all__ = [
    'PATH_TO_DATA',
    'HTML_CODE',
    'SKETCH_IMG',
    'IMAGE_SIZE',
]


PATH_TO_DATA = '/Users/victor/Desktop/sketchy-code/data/sketches/'
HTML_CODE = glob.glob(PATH_TO_DATA + '*.html')
SKETCH_IMG = glob.glob(PATH_TO_DATA + '*.png')
IMAGE_SIZE = (256, 256, 3)

# assert len(HTML_CODE) == len(SKETCH_IMG)
