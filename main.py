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

    ret_y=[]
    ret_x=[]

    for i in range(2,9):

        # 画像の読み込み
        img=cv2.imread(file)
        k=i
        ret, kimg, kimg_s=imcon.Kmeans(img,k)
        ret_y+=[int(ret)]
        ret_x+=[i]

        #出力
        cv2.imwrite("output/{}_k={}.jpg".format(name,k),kimg)
        cv2.imwrite("output/{}_k={}_s.jpg".format(name,k),kimg_s)
    
    fig=plt.figure()
    plt.plot(ret_x,ret_y)
    plt.grid()
    fig.savefig("output/{}_glaph.jpg".format(name))

    a=(ret_y[6]-ret_y[0])/2
    b=ret_y[0]-a

    ans=2
    seg=0

    for i in range(1,6):
        temp=a*np.log2(ret_x[i])+b-ret_y[i]
        if seg<temp:
            ans=i+2
            seg=temp

    print("best image for {} is k={}".format(name,ans))