import time
from time import time as timer
import cv2
import numpy as np
import os

from utils import resize_with_max_ratio
from utils import draw_boxes, scale_bbox, NMS
from utils import calculate_features_image, calculate_features
from utils import sliding_window

import joblib



def visualize_sliding_window(image, box):
    clone = image.copy()
    color = (0, 255, 0)
    cv2.rectangle(clone, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
    cv2.imshow("Window", image)
    cv2.waitKey(1)

def pre_process(image):

    MAX_H, MAX_W = (960, 960)
    MIN_H, MIN_W = (512, 512)

    image = resize_with_max_ratio(image, MAX_H, MAX_W)
    #image = resize_with_min_ratio(image, MIN_H, MIN_W)

    scales = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]

    output = image.copy()
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    return image, output, scales

def face_detection(image, scales):
    patches = []
    bounding_boxes = []

    # loop over the image pyramid
    for scale in scales:

        # loop over the sliding window for each layer of the pyramid
        h = int(image.shape[0] / scale)
        w = int(image.shape[1] / scale)
        resized = cv2.resize(image, (w, h))

        for (x, y, window) in sliding_window(resized, stepSize=8, windowSize=(winW, winH)): #8, 16, 24
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # check if the window is outside our image boundary
            if x + winW > resized.shape[1] or y + winH > resized.shape[0]:
                continue

            box = [x, y, winW, winH]
            box_rescaled = scale_bbox(box, scale)

            bounding_boxes.append(box_rescaled)
            patches.append(window)

    return patches, bounding_boxes

def face_classification(faces, single_image=False, classification_threshold=0.3):
    faces = np.asarray(faces)
    print(faces.shape)

    if single_image:
        features = calculate_features_image(faces).reshape(1, -1)
    else:
        features = calculate_features(faces)

    # scale features
    if std_scaler != None:
        features = std_scaler.transform(features)
    if pca_scaler != None:
        features = pca_scaler.transform(features)

    # predict if head or not
    prob = model.decision_function(features) > classification_threshold
    return prob.astype(np.uint32) #model.predict(features)

def face_detection_pipeline(image, clf_threshold=0.3, nms_threshold=0.1, out_path=None, write=False):
    # preprocess the image
    image, output, scales = pre_process(image)

    t_start = timer()
    print("starting sliding window ...")

    patches, bounding_boxes = face_detection(image, scales)

    t_end = timer() - t_start
    print("sliding window time: {}".format(t_end))

    t_start = timer()
    print("starting classification of patches ...")

    results = face_classification(np.asarray(patches), clf_threshold)

    t_end = timer() - t_start
    print("classification took: {}".format(t_end))

    # Keep only bounding boxes with faces
    bboxs = [bounding_boxes[i] for i in range(0, len(results)) if results[i] == 1]
    bboxs = np.asarray(bboxs)

    # Non Maximum Suppression
    bboxs, _ = NMS(bboxs, threshold=nms_threshold)

    # Remove padding
    image = image[+8:-8, +8:-8]

    # Draw bounding boxes on faces
    draw_boxes(output, bboxs)

    if write:
        cv2.imwrite(out_path, output)




###########################################################################################

pad = 16
target_size = (64, 64)
(winW, winH) = target_size

# load scalers and the model
std_scaler, pca_scaler, model = joblib.load("model.sav")

in_folder = '/home/lfx/Desktop/Recent Projects/Deep Face Detection/test_images/' #'data-path' 
out_folder = './output'

# apply face detection to multiple images
for filename in os.listdir(in_folder):
    image = cv2.imread(os.path.join(in_folder, filename))
    face_detection_pipeline(image, clf_threshold=0.3, nms_threshold=0.1, out_path=os.path.join(out_folder, filename), write=True) 

###########################################################################################
