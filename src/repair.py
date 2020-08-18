import cv2
import numpy as np

def keep_max_cc(img):
    img = img.astype('uint8')
    _, labels_im, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
    max_area = np.argmax(stats[1:, -1]) + 1
    return (labels_im == max_area).astype('i')

def repair(img):
    return keep_max_cc(img)