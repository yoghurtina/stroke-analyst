
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import numpy as np

def open_image(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image

####CLUSTERING###
import matplotlib.pyplot as plt
import cv2

# image = open_image(path)

def clustering_image(img):
    img = open_image(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # reshape the img to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # print(pixel_values.shape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.6)
    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_img = centers[labels.flatten()]

    # reshape back to the original img dimension
    segmented_img = segmented_img.reshape(img.shape)
    # show the img
    plt.imshow(segmented_img)
    plt.show()

    return segmented_img

img = "segmented_6.png"
# img = "code/test.jpg"
clust = clustering_image(img)