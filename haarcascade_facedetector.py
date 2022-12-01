import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from utils import draw_boxes

def face_detector(image):
    boxes = []
    gray = image_original
    if len(image_original.shape) > 2:
        gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    eq = cv2.equalizeHist(gray)
    boxes = face_cascade.detectMultiScale(gray,
                                    scaleFactor=1.3, 
                                    minNeighbors=3, 
                                    minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE)

    return boxes

in_folder = 'input-data-path'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

out_folder = 'output-data-path'
for filename in os.listdir(in_folder):
    image_original = cv2.imread(os.path.join(in_folder, filename))
    print('image size: {}'.format(image_original.shape))

    boxes = face_detector(image_original)
    draw_boxes(image_original, boxes)

    cv2.imwrite(os.path.join(out_folder, filename), image_original)
