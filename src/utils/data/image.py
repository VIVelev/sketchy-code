import cv2

__all__ = [
    'get_preprocessed_img',
    'show',
]


def get_preprocessed_img(img_path, image_size=(256, 256)):
    img = cv2.imread(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, image_size)
    img = img.astype('float32')
    img /= 255

    return img

def show(img_path):
    image = cv2.imread(img_path)
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('view', image)
    cv2.waitKey(0)
    cv2.destroyWindow('view')
