import os
import cv2 as cv
import numpy as np
def calmeanstd(height,width):
    image_num_mean=0
    image_num_std=0
    B_means=0.0
    G_means=0.0
    R_means=0.0
    B_stds=0.0
    G_stds=0.0
    R_stds=0.0
    index_mean=0
    index_std=0
    root_path="E:/pipeline/pipeimages/"
    subroot_names=os.listdir(root_path)
    for subroot_name in subroot_names:
        if index_mean==7:
            break
        # print(subroot_name)
        file_path=os.path.join(root_path,subroot_name)
        # print(file_path,len(file_path))
        image_names=os.listdir(file_path)
        image_num_mean+=len(image_names)
        # print(image_num_mean)
        for image_name in image_names:
            # print(image_name)
            image_path=os.path.join(file_path,image_name)
            # print(image_path)
            image=cv.imread(image_path)
            image = cv.resize(image, (height, width))
            image = image.astype(np.float32) / 255.0
            B_means += np.sum(image[:, :, 0])
            G_means += np.sum(image[:, :, 1])
            R_means += np.sum(image[:, :, 2])
            # print(image)
            # print("mean: ", B_means, G_means, R_means)
        # print(file_path,"means : ", B_means, G_means, R_means)
        index_mean+=1
    B_mean = B_means / image_num_mean / height / width
    G_mean = G_means / image_num_mean / height / width
    R_mean = R_means / image_num_mean / height / width
    print(B_mean,G_mean,R_mean)
    for subroot_name in subroot_names:
        if index_std==7:
            break
        # print(subroot_name)
        file_path=os.path.join(root_path,subroot_name)
        # print(file_path,len(file_path))
        image_names=os.listdir(file_path)
        image_num_std+=len(image_names)
        # print(image_num_std)
        for image_name in image_names:
            # print(image_name)
            image_path=os.path.join(file_path,image_name)
            # print(image_path)
            image=cv.imread(image_path)
            image = cv.resize(image, (height, width))
            image = image.astype(np.float32) / 255.0
            B_stds += np.sum((image[:, :, 0] - B_mean) ** 2)
            G_stds += np.sum((image[:, :, 1] - G_mean) ** 2)
            R_stds += np.sum((image[:, :, 2] - R_mean) ** 2)
        # print(file_path, "stds : ", B_stds, G_stds, R_stds)
        index_std+=1
    B_std = np.sqrt(B_stds / image_num_std / height / width)
    G_std = np.sqrt(G_stds / image_num_std / height / width)
    R_std = np.sqrt(R_stds / image_num_std / height / width)
    print(B_std,G_std,R_std)
    return B_mean,G_mean,R_mean,B_std,G_std,R_std

if __name__=="__main__":
    B_mean,G_mean,R_mean,B_std,G_std,R_std=calmeanstd(200,200)
    # print(B_mean,G_mean,R_mean,B_std,G_std,R_std)