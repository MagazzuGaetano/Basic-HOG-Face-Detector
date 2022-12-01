import os
import cv2
import numpy as np



input_data_folder = 'dtd/images'
output_data_folder = './output-folder'
images = []

for subdir, dirs, files in os.walk(input_data_folder):
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        output_path = os.path.join(output_data_folder, file)

        print(output_path)
        cv2.imwrite(output_path, image)
