import cv2

__all__ = [
    'pencil_sketch',
]


def pencil_sketch(path_to_img, width, height, bg_gray='./preprocess/pencilsketch_bg.jpg'):
    """Pencil sketch effect

    Applies a pencil sketch effect to an image.
    The processed image is overlayed over a background image for visual effect.
    """

    img_rgb = cv2.imread(path_to_img)
    img_rgb = cv2.resize(img_rgb, (width, height))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256)

    canvas = cv2.imread(bg_gray, cv2.CV_8UC1)
    if canvas is not None:
        canvas = cv2.resize(canvas, (width, height))
        img_blend = cv2.multiply(img_blend, canvas, scale=1. / 256)

    return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)
