import cv2

__all__ = [
    'get_preprocessed_img',
    'show',
]


def get_preprocessed_img(img_path, image_size=(256, 256)):
    img_rgb = cv2.imread(img_path)
    img_rgb = cv2.resize(img_rgb, image_size)
    return img_rgb

def show(img_path):
    image = cv2.imread(img_path)
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('view', image)
    cv2.waitKey(0)
    cv2.destroyWindow('view')
