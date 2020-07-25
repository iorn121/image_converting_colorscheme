import numpy as np
import cv2
import image_converting as imcon
from pathlib import Path
import os, glob
import matplotlib.pyplot as plt

# ファイル取得
files = glob.glob("./input/*.jpg")

for file in files:
    name = Path(file).stem
    path_w = "output/result.txt"

    # 画像の読み込み
    img=cv2.imread(file)
    k=5
    kimg, kimg_s=imcon.Kmeans(img,k)

    #出力
    cv2.imwrite("output/{}_k={}.jpg".format(name,k),kimg)
    cv2.imwrite("output/{}_k={}_s.jpg".format(name,k),kimg_s)