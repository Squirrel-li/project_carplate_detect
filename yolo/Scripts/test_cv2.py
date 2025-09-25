import os
import cv2
from pathlib import Path

def main():
    cam_num = 0
    cnt = 0
    
    path_project_root = os.getcwd()
    path_dir_img_cut = os.path.join(path_project_root, "images", "carplates")
    path_dir_img_preprocess = os.path.join(path_project_root, "images", "preprocess_out")

    for i in range(1):
        #for t in range(0, 255):
        path_file_img_cut = os.path.join(path_dir_img_cut, f"{cam_num}_{i}.jpg")
        path_file_img_preprocess = os.path.join(path_dir_img_preprocess, f"{cam_num}_{i}.jpg")#_{t}
        print(path_file_img_cut)
        print(path_file_img_preprocess)
        img = cv2.imread(path_file_img_cut)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(img_gray,42,255,cv2.THRESH_BINARY)
        print(binary)
        cv2.imwrite(path_file_img_preprocess, binary)
    
if __name__ == '__main__':
    main()
