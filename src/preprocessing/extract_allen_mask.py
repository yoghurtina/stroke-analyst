import numpy as np

def extract_allen_mask(index):
    size = [528, 320, 456]
    # VOL = 3-D matrix (volume) of atlas Nissl
    with open('annotation.raw', 'rb') as file:
        VOL = np.fromfile(file, dtype=np.uint8, count=np.prod(size))
    VOL = np.reshape(VOL, size)

    # Reshape the numpy array to the desired size
    VOL = VOL.reshape((size[2], size[1], size[0])) # swap the order of dimensions
    VOL = np.transpose(VOL, (2, 1, 0))  # swap dimensions to (456, 320, 528)

    mask = []
    for i in range(size[0]):
        mask.append(VOL[i, :, :])
    slice = mask[index]
    
    return slice


