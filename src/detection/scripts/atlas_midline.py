import cv2
import numpy as np

def compute_middle_line(image, h, w):
    """
    Compute the middle line (ML) of a coronal section image using SSD.

    Args:
        image (numpy.ndarray): Binary image representation of the coronal section.
        h (int): Limit of height, equal to the height of each section.
        w (int): Limit of thickness, 1/5 of the section thickness.

    Returns:
        middle_line (numpy.ndarray): Vertical line representing the ML.
    """
    height, width = image.shape[:2]
    box_width = h // 2
    box_height = w // 2
    min_ssd = np.inf
    best_indicator = 0

    for i in range(width - box_width):
        box1 = image[:, i:i+box_width]
        # Pad box2 to match the dimensions of box1
        box2 = np.flip(image[:, i+box_width-1:i+box_width+box_width-1], axis=1)
        ssd = np.sum((box1 - box2) ** 2)
        if ssd < min_ssd:
            min_ssd = ssd
            best_indicator = i

    middle_line = np.array([[best_indicator + box_width, 0],
                            [best_indicator + box_width, height]], dtype=np.int32)

    return middle_line



# Example usage
# Load binary image of coronal section using OpenCV
image = cv2.imread('sq.png', 0)
# Convert to binary (0, 1) representation
image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]
# Compute middle line
middle_line = compute_middle_line(image, h=image.shape[0], w=image.shape[1])
# Draw middle line on original image
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.polylines(result, [middle_line], False, (0, 255, 0), 2)
cv2.imshow('Middle Line', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
