import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

from skimage.feature import local_binary_pattern  # , hog
from cyvlfeat.hog import hog

import random



####################################################################################################################################################


def draw_boxes(image, boxes):
    for box in boxes:
        x, y, W, H = box
        start_point = (x, y)
        end_point = (x + W, y + H)
        color = (0, 255, 0)
        cv2.rectangle(image, start_point, end_point, color, 2)


def scale_bbox(box, scale):
    x, y, w, h = np.asarray(box).astype(np.float64)

    x = int(x * scale)
    w = int(w * scale)
    y = int(y * scale)
    h = int(h * scale)

    """
    x = int(x / (new_size[0]/orig_size[0]))
    w = int(w / (new_size[0]/orig_size[0]))
    y = int(y / (new_size[1]/orig_size[1]))
    h = int(h / (new_size[1]/orig_size[1]))
    """
    return [x, y, w, h]


def NMS(boxes, threshold=0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return [], []

    # From [x, y, w, h] to [x1, y1, x2, y2] top (x1,y1) and bottom (x2,y2)
    boxes = np.asarray([[x, y, w + x - 1, h + y - 1] for x, y, w, h in boxes])

    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner

    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (
        y2 - y1 + 1
    )  # We add 1, because the pixel at the start as well as at the end counts

    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):

        # Create temporary indices
        temp_indices = indices[indices != i]

        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices, 0])
        yy1 = np.maximum(box[1], boxes[temp_indices, 1])
        xx2 = np.minimum(box[2], boxes[temp_indices, 2])
        yy2 = np.minimum(box[3], boxes[temp_indices, 3])

        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]

        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
        if np.any(overlap > threshold):
            indices = indices[indices != i]
            #the current bounding box is the mean of the other that overlap
            box[0] = int(np.mean(boxes[temp_indices, 0]))
            box[1] = int(np.mean(boxes[temp_indices, 1]))
            box[2] = int(np.mean(boxes[temp_indices, 2]))
            box[3] = int(np.mean(boxes[temp_indices, 3]))

    # return only the boxes at the remaining indices
    boxes = boxes[indices].astype(int)

    # From [x1, y1, x2, y2] top (x1,y1) and bottom (x2,y2) to [x, y, w, h]
    boxes = np.asarray([[x1, y1, x2 - x1 + 1, y2 - y1 + 1] for x1, y1, x2, y2 in boxes])

    return boxes, indices


def resize_with_max_ratio(image, max_h, max_w):
    w, h, ch = image.shape
    if (h > max_h) or (w > max_w):
        rate = max_h / h
        rate_w = w * rate
        if rate_w > max_h:
            rate = max_h / w
        image = cv2.resize(
            image, (int(h * rate), int(w * rate)), interpolation=cv2.INTER_CUBIC
        )
    return image


def resize_with_min_ratio(image, min_h, min_w):
    w, h, ch = image.shape
    if (h > min_h) or (w > min_w):
        rate = min_h / h
        rate_w = w * rate
        if rate_w < min_h:
            rate = min_h / w
        image = cv2.resize(
            image, (int(h * rate), int(w * rate)), interpolation=cv2.INTER_CUBIC
        )
    return image


def random_crop(img, height, width):
    x = random.randint(0, img.shape[1] - int(width))
    y = random.randint(0, img.shape[0] - int(height))
    cropped = img[y : y + height, x : x + width]
    return cropped


def crop_center(image, w, h):
    x = image.shape[1] / 2 - w / 2
    y = image.shape[0] / 2 - h / 2

    crop_img = image[int(y) : int(y + h), int(x) : int(x + w)]
    return crop_img


####################################################################################################################################################


def calculate_hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=False,
):
    # 9, 16, 2
    fd, hog_image = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
    )

    if visualize:
        plt.imshow(hog_image, cmap="gray")
        plt.title("hog")
        plt.show()

    return fd


def calculate_lbp(image, radius=2, n_points=None, method="default"):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if n_points == None:
        n_points = 8 * radius

    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = local_binary_pattern(image, n_points, radius, method)
    (hist, _) = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )

    eps = 1e-7
    # normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum() + eps

    return hist


def calculate_features_image(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hog_hist = hog(gray, n_orientations=9, cell_size=8).flatten()

    # lbp_hist = calculate_lbp(gray, radius=2, n_points=8, method='uniform')
    # lbp_histR = calculate_lbp(image[:,:,0], radius=2, n_points=8, method='uniform')
    # lbp_histG = calculate_lbp(image[:,:,1], radius=2, n_points=8, method='uniform')
    # lbp_histB = calculate_lbp(image[:,:,2], radius=2, n_points=8, method='uniform')
    # features = np.hstack([lbp_hist, hog_hist])

    return hog_hist


def calculate_features(images):
    list_features = []
    for i in range(images.shape[0]):
        if len(images.shape) > 3:
            features = calculate_features_image(images[i, :, :, :])
        else:
            features = calculate_features_image(images[i, :, :])
        list_features.append(features)
    return np.array(list_features)


####################################################################################################################################################



def load_images_from_folder(folder, N, M, Z):
    filenames = os.listdir(folder)

    k = 0
    images = np.zeros((Z, N, M, 3), dtype=np.uint8)
    for filename in filenames:
        if k == Z:
            break

        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:

            if image.shape[0] != M or image.shape[1] != N:
                image = cv2.resize(image, (M, N))
            
            if len(image.shape) > 2:
                images[k, :, :, :] = image
            else:
                images[k, :, :, 0] = image
                images[k, :, :, 1] = image
                images[k, :, :, 2] = image

        k = k + 1

    return images


def pyramid(image, scale=1.5, minSize=(256, 256)):
    # yield the original image
    yield (scale, image)
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield (scale, image)


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])
