import cv2
import numpy as np

__all__ = [
    'get_preprocessed_img',
    'show',
]


def get_preprocessed_img(img_path, image_size=(256, 256)):
    img_rgb = cv2.imread(img_path)
    img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
    img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
    resized = cv2.resize(img_stacked, (200,200), interpolation=cv2.INTER_AREA)
    bg_img = 255 * np.ones(shape=(256,256,3))
    bg_img[27:227, 27:227, :] = resized
    bg_img /= 255
    return bg_img

def show(img_path):
    image = cv2.imread(img_path)
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('view', image)
    cv2.waitKey(0)
    cv2.destroyWindow('view')
