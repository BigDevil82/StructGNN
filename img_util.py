import os

import cv2
import numpy as np
from PIL import Image


def show(img, title="img", wait_sec=0, keep=False):
    # return
    cv2.imshow(title, img)
    if wait_sec >= 0:
        cv2.waitKey(int(wait_sec * 1000))
    if not keep:
        cv2.destroyAllWindows()


def calc_IoU(mask1, mask2):
    """
    calculate intersection over union of two images of mask

    Args:
     - img1: image1 of mask
     - img2: image2 of mask

    Returns:
     IoU value
    """
    # Calculate the intersection and union
    intersection = cv2.bitwise_and(mask1, mask2)
    # show(intersection, "intersection")
    union = cv2.bitwise_or(mask1, mask2)
    # show(union, "union")
    iou = np.sum(intersection) / np.sum(union)

    return iou


def segment_gray(img, lower=150, upper=155):
    """
    create gray mask for the image
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(gray, lower, upper)
    # show(mask, "mask")
    return mask


def segment_by_hsv(img, lower, upper):
    """
    create a mask for some kind of color in the image based on HSV color space

    Args:
     - img: the image to be segmented, in BGR color space
     - lower: the lower hsv value bound for the color, e.g. [0, 70, 50]
       it can be a nested list, where each sublist is a bound hsv
       so multiple inRange filters are combined to form the final result
     - upper: the upper hsv value bound for the color, like lower

     Returns:
        a mask of the color, one channel, 255 if the pixel is in the color range, 0 otherwise
    """

    if not isinstance(lower[0], list):
        lower = [lower]
    if not isinstance(upper[0], list):
        upper = [upper]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in range(len(lower)):
        lower[i] = np.array(lower[i])
        upper[i] = np.array(upper[i])
        mask = cv2.inRange(img, lower[i], upper[i])
        final_mask = cv2.bitwise_or(final_mask, mask)
    return final_mask


def segment_red(img):
    """
    create mask for red color in the image based on HSV color space
    """

    lower = [[0, 70, 50], [170, 70, 50]]
    upper = [[10, 255, 255], [180, 255, 255]]
    mask = segment_by_hsv(img, lower, upper)

    return mask


def resize_img(src, des=None, size=(512, 256)):
    """
    resize the image to the specified size with NEAREST algorithm

    Args:
     - src, the source image path or opencv image
     - des, if specified, the image will be saved to the specified path
     - size, (width, height) of the result image

     returns: the resized image

    """
    assert isinstance(src, (str, np.ndarray))
    img = Image.open(src) if isinstance(src, str) else Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    img = img.resize(size, Image.NEAREST)
    if des is not None:
        if not os.path.exists(os.path.dirname(des)):
            os.makedirs(os.path.dirname(des), exist_ok=True)
        img.save(des)
    # convert PIL image to opencv image
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def calculate_shearwall_IoU(gen, engineer):
    """
    calculate IoU of shearwall from AI generated image and engineer designed image

    Args:
     - gen: path of generated image or opencv image
     - engineer: path of engineer designed image or opencv image

    Return IoU value
    """

    if isinstance(gen, str):
        gen = cv2.imread(gen)
    if isinstance(engineer, str):
        engineer = cv2.imread(engineer)

    gen_sw = segment_red(gen)
    engineer_sw = segment_red(engineer)
    iou = calc_IoU(gen_sw, engineer_sw)

    return iou


def draw_contours(img, contours, save_path=None):
    """
    draw contours on the image and show it
    """

    cv2.draw_contours(img, contours, -1, (0, 255, 0), 2)
    show(img, "contours")
    if save_path is not None:
        cv2.imwrite(save_path, img)


def draw_rects(img, rects, save_path=None):
    """
    draw rects on the image and show it
    """

    img_with_rects = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_with_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show(img_with_rects, "rects")
    if save_path is not None:
        cv2.imwrite(save_path, img_with_rects)


def pad_img(img, pad_size, color=255):
    """
    add padding to the image

    Args:
     - img: the image to be padded
     - pad_size: the size of padding, if a scalar, the same padding will be applied to all sides
       if a tuple, the padding will be applied to each side separately
    - color: the color of the padding area
    """

    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        padded_img = np.zeros((h + pad_size[0] * 2, w + pad_size[1] * 2), dtype=np.uint8)
    else:
        padded_img = np.zeros((h + pad_size[0] * 2, w + pad_size[1] * 2, 3), dtype=np.uint8)
    padded_img.fill(color)
    padded_img[pad_size[0] : h + pad_size[0], pad_size[1] : w + pad_size[1]] = img
    return padded_img


def adjust_bright_contrast(img, alpha=1.0, beta=0):
    """
    adjust the brightness and contrast of the image

    Args:
     - img: the image to be adjusted
     - alpha: the contrast control, 1.0 means no change
     - beta: the brightness control, 0 means no change
    """

    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def gamma_correction(img, gamma=1.0):
    """
    adjust the gamma of the image

    Args:
     - img: the image to be adjusted
     - gamma: the gamma value, 1.0 means no change
    """

    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted = cv2.LUT(img, table)
    return adjusted


def correct_gen_img(img_path):
    """
    archi-diffusion generated layout image may contain some shadows that are not desired
    this function will remove those shadows and only keep wall lines
    """
    img = cv2.imread(img_path)
    corrected = adjust_bright_contrast(img, 1.45, 0)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    corrected = img.copy()
    corrected[gray > 240] = [255, 255, 255]
    return corrected


def make_variations(img_path):
    """
    make three variations of the image:
        - flip the image vertically
        - flip the image horizontally
        - rotate the image 180 degrees
    Save them with the original name with suffix _v, _h, _r
    """
    img = cv2.imread(img_path)
    img_flip_v = cv2.flip(img, 0)
    img_flip_h = cv2.flip(img, 1)
    img_rot = cv2.rotate(img, cv2.ROTATE_180)
    return img, img_flip_v, img_flip_h, img_rot


def rgb2hsv(rgb):
    """
    convert rgb color to hsv color
    """
    bgr = np.array(rgb)[::-1].reshape(1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv.flatten()
