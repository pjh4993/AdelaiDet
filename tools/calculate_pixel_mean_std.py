import os
import numpy as np
import glob
import cv2
from tqdm import tqdm

dataset_root = 'datasets/nlos/images'
init_root = 'datasets/nlos/initialization'

laser_image = []
init_image = []

image_sum = None
squared_sum = None

init_exclude_sum = None
init_squared_exclude_sum = None

count = len(glob.glob(os.path.join(dataset_root, '*')))

for image in glob.glob(os.path.join(init_root, '*')):
    init_image.append(np.expand_dims(cv2.imread(image).astype(np.float64), axis=0))

init_image = np.concatenate(init_image, axis=0)


count = 0
for subdir in tqdm(glob.glob(os.path.join(dataset_root, '*')), desc="dir image loading"):
    dir_image = []
    for image in glob.glob(os.path.join(subdir, '*')):
        if image.find('reflection') == -1:
            continue
        dir_image.append(np.expand_dims(cv2.imread(image).astype(np.float64),axis=0))
    dir_image = np.expand_dims(np.concatenate(dir_image, axis=0), axis=0)
    if type(image_sum) == type(None):
        image_sum = dir_image
        squared_sum = dir_image ** 2
        init_exclude_sum = dir_image - init_image
        init_squared_exclude_sum = (dir_image - init_image) ** 2
    else:
        image_sum += dir_image
        squared_sum += dir_image ** 2
        init_exclude_sum += (dir_image - init_image)
        init_squared_exclude_sum += ((dir_image - init_image) ** 2)
    
    count += 1
    if count > 3:
        break

#laser_image = np.concatenate(laser_image, axis=0)
image_sum = image_sum 
squared_sum = squared_sum 
init_exclude_sum = init_exclude_sum 
init_squared_exclude_sum = init_squared_exclude_sum 

#laser_image = np.transpose(laser_image.reshape(-1, 50 * 530 * 1936, 3), (0,2,1))
image_sum = np.transpose(image_sum.reshape(1, 51304000, 3), (0,2,1))
squared_sum = np.transpose(squared_sum.reshape(1, 51304000, 3), (0,2,1))
init_exclude_sum= np.transpose(init_exclude_sum.reshape(1, 51304000, 3), (0,2,1))
init_squared_exclude_sum= np.transpose(init_squared_exclude_sum.reshape(1, 51304000, 3), (0,2,1))

print('mean: ',(image_sum).mean(axis=2) / count)
print('std:',np.sqrt((squared_sum.mean(axis=2) / count - (image_sum.mean(axis=2)/ count) ** 2)))
print('init exclude mean: ',init_exclude_sum.mean(axis=2) / count)
print('init exclude std:',np.sqrt((init_squared_exclude_sum.mean(axis=2) / count - (init_exclude_sum.mean(axis=2) / count) ** 2)))

std = np.expand_dims(np.sqrt((squared_sum.mean(axis=2) / count - (image_sum.mean(axis=2)/ count) ** 2)), axis=2)
mean = np.expand_dims((image_sum).mean(axis=2) / count, axis=2)
image_sum = image_sum/count 
print(((image_sum - mean) / std).mean(axis=2))