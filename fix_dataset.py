import os
import cv2
import numpy as np



data_path = '/home/lfx/Downloads/dtd/images'
images = []

for subdir, dirs, files in os.walk(data_path):
    for file in files:
        #print(os.path.join(subdir, file))
        image = cv2.imread(os.path.join(subdir, file))
        output_path = os.path.join('/home/lfx/Downloads', 'New Folder', file)

        print(output_path)
        cv2.imwrite(output_path, image)
