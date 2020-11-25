import glob
import os
import shutil

dataset_root = "/media/pjh3974/datasets/NLOS/nlos/images"
target_root = "/media/pjh3974/datasets/NLOS/nlos/annotations/xml"

for img_dir in glob.glob(os.path.join(dataset_root, '*')):
    file_name  = img_dir.split('/')[-1]
    if file_name == 'initialization':
        continue
    shutil.copyfile(src=os.path.join(img_dir, 'gt_rgb_image.png'), dst=os.path.join(target_root, file_name + '.png'))
 