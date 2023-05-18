import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import numpy as np
import base64
from PIL import Image


def find_mask(image):
    CHECKPOINT_PATH = "/home/ioanna/Documents/Thesis/src/sam_weights/sam_vit_b_01ec64.pth"
    DEVICE = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_b"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)

    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam_result = mask_generator.generate(image_rgb)
    
    # print(sam_result[0].keys())

    mask_annotator = sv.MaskAnnotator()

    detections = sv.Detections.from_sam(sam_result=sam_result)

    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    masks = [
    mask['segmentation']
    for mask
    in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]

    sv.plot_images_grid(
    images=masks,
    grid_size=(8, 8),
    size=(16, 16)
)   

    return True

find_mask("moving.jpg")