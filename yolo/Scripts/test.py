import os
import cv2
from pathlib import Path
from ultralytics import YOLO
#from paddleocr import PaddleOCR
#import paddlelite.lite


#import camera_ctrl
#import camera_detect
import car_plates_detect

if __name__ == '__main__':
    IP_NUM = "10.244.18.107"
    cam_num = 0
    cnt = 0
    
    #define path
    abs_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(abs_path)
    path_project_root = os.path.join(dir_path, "..")
    path_project_root = os.path.normpath(path_project_root)
    path_dir_images = os.path.join(path_project_root, "images")

    path_dir_img_cut = os.path.join(path_dir_images, "carplates")
    path_dir_img_preprocess = os.path.join(path_dir_images, "preprocess_out")
    path_dir_text_ocr = os.path.join(path_dir_images, "OCR_out")
    
    ncnn_model = YOLO("best_ncnn_model")

    print("begin")

    for i in range(8):
        cam_num = i
        #detect begin
        path_file_img_input = os.path.join(path_dir_images, "input", f'{cam_num}.jpg')
        yolo_result = car_plates_detect.detect(path_file_img_input, ncnn_model)
        boxes = yolo_result[0].boxes.xyxy
        print("detect done. boxes position at:")
        cnt = 0
        for box in boxes:
            print(f"box{cnt} at {box}")
            cnt += 1
        print()

        #cut images
        cnt = 0
        for box in boxes:
            path_file_img_cut = os.path.join(path_dir_img_cut, f"{cam_num}_{cnt}.jpg")
            car_plates_detect.cut_images(path_file_img_input, path_file_img_cut, box, cam_num)
            cnt += 1
        print(f"cut done. put {cnt} images at {path_dir_img_cut}\n")
        

        #preprocess
        car_plates_num = len(boxes)
        for cnt in range(car_plates_num):
            path_file_img_cut = os.path.join(path_dir_img_cut, f"{cam_num}_{cnt}.jpg")
            path_file_img_preprocess = os.path.join(path_dir_img_preprocess, f"{cam_num}_{cnt}.jpg")
            car_plates_detect.preprocess(path_file_img_cut, path_file_img_preprocess, 2)
        print(f"preprocess done. put {cnt} images at {path_dir_img_preprocess}\n")