import os
import cv2
from pathlib import Path
from ultralytics import YOLO
#from paddleocr import PaddleOCR
#from paddlelite.lite


#import camera_ctrl
#import camera_detect

def detect(path_input, ncnn_model):
    results = ncnn_model(path_input)
    return results

def cut_images(path_input, path_output, box, CAM_NUM):
    cnt = 0
    img = cv2.imread(path_input)
    x1, y1, x2, y2 = [int(coord) for coord in box]
    cropped_img = img[y1:y2, x1:x2]
    (h, w) = cropped_img.shape[:2]
    ratio = 160 / float(h)
    target_width = int(w * ratio)
    new_dimensions = (target_width, 160)
    resized_image = cv2.resize(cropped_img, new_dimensions, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path_output, resized_image)


def preprocess(path_input, path_output, mod):
    '''
    mod = 0: gray
    mod = 1: gray + denoised
    mod = 2: gray + denoised + denoise
    '''
    img = cv2.imread(path_input)
    '''
    if mod >= 0:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img_gray
    if mod >= 1:
        denoised = cv2.medianBlur(img_gray, 3)
        result = denoised
    if mod >= 2:'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(img_gray)
    ret, binary = cv2.threshold(dst,42,255,cv2.THRESH_BINARY)
    result = binary
    cv2.imwrite(path_output, result)