import numpy as np
import cv2
import matplotlib.pyplot as plt
from cyvlfeat.hog import hog
from utils import calculate_hog, random_crop


image_path = "image-path"

image = cv2.imread(image_path)

print(image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")
plt.title("gray")
plt.show()

h, w = gray.shape
hog_hist = hog(gray, n_orientations=9, cell_size=8)
#hog_hist = calculate_hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

print(hog_hist.flatten().shape)
