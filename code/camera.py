# image.shape = [480,640,3]

import cv2 as cv
import os
import numpy as np

root_path="E:/pipeline/pipeimages/"
file_names=os.listdir(root_path)
indexes=[0,0,0,0]
for i in range(len(file_names)):
    subroot_path=os.path.join(root_path,str(i))
    image_numbers=len(os.listdir(subroot_path))
    indexes[i]=image_numbers+1
index0,index1,index2,index3=indexes[0],indexes[1],indexes[2],indexes[3]
cap=cv.VideoCapture(0)
while True:
    ret,image_raw=cap.read()
    if ret is not True:
        break
    image=image_raw[:,80:560,:]
    # print(image.shape)
    cv.imshow("raw",image)
    ch=cv.waitKey(1)
    if ch==48:
        cv.imwrite("E:/pipeline/pipeimages/0/0_Type_"+str(index0)+".jpg",image)
        blob=cv.imread("E:/pipeline/pipeimages/0/0_Type_"+str(index0)+".jpg")
        cv.imshow("blob",blob)
        ca=cv.waitKey(1)
        while ca!=111 and ca!=110:
            ca=cv.waitKey(1)
        if ca==111:
            index0+=1
            cv.destroyWindow("blob")
            print("已经存储照片"+str(index0))
        elif ca==110:
            cv.destroyWindow("blob")
            print("重新拍照。")
    if ch==49:
        cv.imwrite("E:/pipeline/pipeimages/1/1_Type_"+str(index1)+".jpg",image)
        blob=cv.imread("E:/pipeline/pipeimages/1/1_Type_"+str(index1)+".jpg")
        cv.imshow("blob",blob)
        ca=cv.waitKey(1)
        while ca!=111 and ca!=110:
            ca=cv.waitKey(1)
        if ca==111:
            index1+=1
            cv.destroyWindow("blob")
            print("已经存储照片"+str(index1))
        elif ca==110:
            cv.destroyWindow("blob")
            print("重新拍照。")
    if ch==50:
        cv.imwrite("E:/pipeline/pipeimages/2/2_A_Type_"+str(index2)+".jpg",image)
        blob=cv.imread("E:/pipeline/pipeimages/2/2_A_Type_"+str(index2)+".jpg")
        cv.imshow("blob",blob)
        ca=cv.waitKey(1)
        while ca!=111 and ca!=110:
            ca=cv.waitKey(1)
        if ca==111:
            index2+=1
            cv.destroyWindow("blob")
            print("已经存储照片"+str(index2))
        elif ca==110:
            cv.destroyWindow("blob")
            print("重新拍照。")
    if ch==51:
        cv.imwrite("E:/pipeline/pipeimages/3/2_B_Type_"+str(index3)+".jpg",image)
        blob=cv.imread("E:/pipeline/pipeimages/3/2_B_Type_"+str(index3)+".jpg")
        cv.imshow("blob",blob)
        ca=cv.waitKey(1)
        while ca!=111 and ca!=110:
            ca=cv.waitKey(1)
        if ca==111:
            index3+=1
            cv.destroyWindow("blob")
            print("已经存储照片"+str(index3))
        elif ca==110:
            cv.destroyWindow("blob")
            print("重新拍照。")
    if ch==27:
        cv.destroyAllWindows()
        break
