import cv2
import numpy as np
import matplotlib.pyplot as plt

from utilities import *

# image_preprocessing demo
image = cv2.imread("dataset/intersection/intersection (2).jpg")
preprocessed_image = image_preprocessing(image)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original image")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("Preprocessed image")
plt.imshow(preprocessed_image)
plt.savefig("media/image_preprocessing.png")

# direction demo
image1 = cv2.imread("dataset/single_line/single_line (2).jpg")
image2 = cv2.imread("dataset/single_line/single_line (5).jpg")
cx1 = direction(image1)
cx2 = direction(image2)
image1 = cv2.circle(image1, (int(cx1), int(
    image1.shape[0]/2)), 50, (0, 0, 255), -1)
image2 = cv2.circle(image2, (int(cx2), int(
    image2.shape[0]/2)), 50, (0, 0, 255), -1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Left turn")
plt.imshow(image1)
plt.subplot(1, 2, 2)
plt.title("Right turn")
plt.imshow(image2)
plt.savefig("media/direction.png")

# turn_detection demo
image1 = cv2.imread("dataset/intersection/intersection (4).jpg")
image2 = cv2.imread("dataset/intersection/intersection (14).jpg")
turn1 = turn_detection(image1)
turn2 = turn_detection(image2)


def turn_to_result(turn):
    if turn == [1, 1]:
        result = "Left and Right turns"
    elif turn == [1, 0]:
        result = "Left turn"
    elif turn == [0, 1]:
        result = "Right turn"
    else:
        result = "No turn"
    return result


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title(turn_to_result(turn1))
plt.imshow(image1)
plt.subplot(1, 2, 2)
plt.title(turn_to_result(turn2))
plt.imshow(image2)
plt.savefig("media/turn_detection.png")

# detect_red demo
image1 = cv2.imread("dataset/env_3_marker/env_3_marker (2).jpg")
image2 = cv2.imread("dataset/intersection/intersection (4).jpg")
red1 = detect_red(image1)
red2 = detect_red(image2)
if red1 == True:
    red1 = "Red"
else:
    red1 = "No red"
if red2 == True:
    red2 = "Red"
else:
    red2 = "No red"
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title(red1)
plt.imshow(image1[:, :, ::-1])
plt.subplot(1, 2, 2)
plt.title(red2)
plt.imshow(image2)
plt.savefig("media/detect_red.png")
