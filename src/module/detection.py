import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import supervision as sv
import numpy as np
import base64
from PIL import Image

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    print(encoded)
    return encoded

def seg_anything(image, bbox):
    CHECKPOINT_PATH = "../sam_weights/sam_vit_h_4b8939.pth"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    # image = encode_image(image)
    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor.set_image(image_rgb)

    box = np.array([
    bbox['x'], 
    bbox['y'], 
    bbox['x'] + bbox['width'], 
    bbox['y'] + bbox['height']
    ])

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
    )

    return source_image, segmented_image

bbox = {'x': 9, 'y': 40, 'width': 330, 'height': 412}
box = np.array([
    bbox['x'], 
    bbox['y'], 
    bbox['x'] + bbox['width'], 
    bbox['y'] + bbox['height']
    ])

print(box)
# encode_image('moving.jpg')
seg_anything('moving.jpg', box)

