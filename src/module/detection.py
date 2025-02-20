import cv2
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageChops
from scipy.ndimage import gaussian_filter
import torch
from transformers import SamModel, SamProcessor, SamConfig

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    print(encoded)
    return encoded

def seg_anything(image, bbox):
    CHECKPOINT_PATH = "src/sam_weights/stroke_huge_1200_batch8.pth"
    DEVICE = 'cpu'
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    # CHECKPOINT_PATH = "src/sam_weights/stroke_huge.pth"

    mask_predictor = SamPredictor(sam)
    image_bgr = cv2.imread(image)
    im_array = np.asarray(image_bgr)

    uploaded = Image.open(image)

    im_width, im_height = im_array.shape[1], im_array.shape[0]

    print(im_width, im_height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor.set_image(image_rgb)

    if im_array is not None:
        box = np.array([
            bbox['x'], 
            bbox['y'], 
            bbox['x'] + bbox['width'], 
            bbox['y'] + bbox['height']
        ])

        # Corrected box2 definition
        box2 = np.array([
            bbox['x'] + bbox['width'],  # Start from the right side of box1
            bbox['y'],                   # Same top edge as box1
            im_width,                    # Right edge of the image
            bbox['y'] + bbox['height']   # Same bottom edge as box1
        ])

        print(box)
        print(box2)


        masks_hem1, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        box_annotator = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.INDEX)
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX) # INDEX # CLASS #TRACK
        
        masks_hem2, scores2, logits2 = mask_predictor.predict(
            box=box2,
            multimask_output=True
        )
        box_annotator2 = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.INDEX)
        mask_annotator2 = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX) # INDEX # CLASS #TRACK

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks_hem1),
            mask=masks_hem1
        )
        detections = detections[detections.area == np.max(detections.area)]

        detections2 = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks_hem2),
            mask=masks_hem2
        )
        detections2 = detections2[detections2.area == np.max(detections2.area)]

        source_image_hem1 = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image_hem1 = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        source_image_hem2 = box_annotator2.annotate(scene=image_bgr.copy(), detections=detections2)
        segmented_image_hem2 = mask_annotator2.annotate(scene=image_bgr.copy(), detections=detections2)

    source_image_array = Image.fromarray(source_image_hem1)
    source_image_array.save("results/detection/source_image_hem1.jpg")

    seg_image_array = Image.fromarray(segmented_image_hem1)
    seg_image_array.save("results/detection/segmented_image_hem1.jpg")
    
    mask_array = Image.fromarray(masks_hem1[0])
    mask_array.save("results/detection/mask1_hem1.jpg")

    mask_array = Image.fromarray(masks_hem1[1])
    mask_array.save("results/detection/mask2_hem1.jpg")

    mask_array = Image.fromarray(masks_hem1[2])
    mask_array.save("results/detection/mask3_hem1.jpg")

    source_image_array = Image.fromarray(source_image_hem2)
    source_image_array.save("results/detection/source_image_hem2.jpg")

    seg_image_array = Image.fromarray(segmented_image_hem2)
    seg_image_array.save("results/detection/segmented_image_hem2.jpg")
    
    mask_array = Image.fromarray(masks_hem2[0])
    mask_array.save("results/detection/mask1_hem2.jpg")

    mask_array = Image.fromarray(masks_hem2[1])
    mask_array.save("results/detection/mask2_hem2.jpg")

    mask_array = Image.fromarray(masks_hem2[2])
    mask_array.save("results/detection/mask3_hem2.jpg")


    return True


def seg_anything_bgs(image, bbox):

    CHECKPOINT_PATH = "src/sam_weights/bgs_huge_1200_batch8.pth"
    MODEL_TYPE = "vit_h"

    # Load model with checkpoint, ensuring CPU compatibility
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    sam = sam_model_registry[MODEL_TYPE]()
    sam.load_state_dict(checkpoint)

    mask_predictor = SamPredictor(sam)

    # Load and process image
    image_bgr = cv2.imread(image)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image}")

    im_array = np.asarray(image_bgr)
    im_width, im_height = im_array.shape[1], im_array.shape[0]  # Correct width & height

    print(im_width, im_height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor.set_image(image_rgb)

    if im_array is not None:
        box = np.array([
            bbox['x'], 
            bbox['y'], 
            bbox['x'] + bbox['width'], 
            bbox['y'] + bbox['height']
        ])

        print("Bounding Box:", box)

        masks_bgs, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )

        box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks_bgs),
            mask=masks_bgs
        )
        detections = detections[detections.area == np.max(detections.area)]

        source_image_annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image_annotated = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    # Save results
    Image.fromarray(source_image_annotated).save("results/segmentation/source_image.jpg")
    Image.fromarray(segmented_image_annotated).save("results/segmentation/segmented_image.jpg")

    for i, mask in enumerate(masks_bgs):
        mask_array = Image.fromarray(mask)
        mask_array.save(f"results/segmentation/mask_{i+1}_bgs.jpg")

    return True


# def seg_anything_bgs(image_path, bbox):
#     CHECKPOINT_PATH = "src/sam_weights/bgs_huge_1200_batch8_og.pth"

#     # Load SAM processor
#     processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

#     # Load SAM model configuration
#     config = SamConfig.from_pretrained("facebook/sam-vit-huge")

#     # Initialize model
#     model = SamModel(config)

#     # Load model weights from checkpoint
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
#     model.load_state_dict(checkpoint)

#     # Set model to evaluation mode
#     model.eval()

#     # Load and process image
#     image_bgr = cv2.imread(image_path)
#     if image_bgr is None:
#         raise ValueError(f"Failed to load image: {image_path}")

#     # Get original image dimensions
#     height, width = image_bgr.shape[:2]

#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(image_rgb)

#     # Process input image
#     inputs = processor(images=pil_image, return_tensors="pt")

#     # Prepare bounding box with shape (batch_size, nb_boxes, 4)
#     # Make sure these coordinates are in [x_min, y_min, x_max, y_max] format
#     box = torch.tensor([[
#         [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']]
#     ]]).float()

#     # Run model inference
#     with torch.no_grad():
#         outputs = model(**inputs, input_boxes=box)

#     # Extract segmentation masks from model outputs.
#     masks_bgs = outputs.pred_masks.cpu().numpy()
    
#     # We’ll store all final single-channel masks (resized to original image dims)
#     mask_list = []
#     # We’ll store the bounding boxes (xyxy format)
#     xyxy_list = []
    
#     # Ensure we iterate over masks in a consistent way
#     # For a single mask (shape might be (1, 3, H, W)), just wrap it in a list
#     if masks_bgs.shape[0] == 1:
#         masks_iter = [masks_bgs[0]]
#     else:
#         masks_iter = masks_bgs

#     # Process each mask
#     for mask in masks_iter:
#         # Remove extra dimension if present: e.g., (1, 3, H, W)
#         if mask.ndim == 4 and mask.shape[0] == 1:
#             mask = mask[0]  # now shape could be (3, H, W) or (1, H, W)

#         # Convert mask to boolean
#         mask_bool = mask.astype(bool)

#         # Ensure we pass a single-channel 2D mask to sv.mask_to_xyxy()
#         if mask_bool.ndim == 3:
#             # e.g. (3, H, W) or (1, H, W)
#             mask_for_bbox = mask_bool[0]  # shape (H, W)
#         elif mask_bool.ndim == 2:
#             mask_for_bbox = mask_bool     # already (H, W)
#         else:
#             raise ValueError(f"Unsupported mask shape for bbox: {mask_bool.shape}")

#         # Compute bounding box from the single-channel mask
#         xyxy_list.append(sv.mask_to_xyxy(mask_for_bbox))

#         # Resize that single-channel mask to the original image resolution
#         mask_channel = mask_for_bbox.astype(np.uint8)
#         mask_resized = cv2.resize(mask_channel, (width, height), interpolation=cv2.INTER_NEAREST)
#         mask_resized = mask_resized.astype(bool)

#         # If the annotator expects 3D masks (C, H, W), stack into 3 channels
#         mask_resized_3c = np.stack([mask_resized]*3, axis=0)
        
#         mask_list.append(mask_resized_3c)

#     # Stack bounding boxes into a 2D array (N, 4)
#     xyxy_array = np.vstack(xyxy_list)

#     # If there's just one detection, use that single 3D mask; otherwise stack them
#     if len(mask_list) == 1:
#         detection_masks = mask_list[0]
#     else:
#         detection_masks = np.array(mask_list)

#     # Build detections
#     detections = sv.Detections(
#         xyxy=xyxy_array,
#         mask=detection_masks
#     )

#     # (Optional) Filter to keep only the largest-area detection
#     detections = detections[detections.area == np.max(detections.area)]

#     # Annotators for bounding boxes and masks
#     box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
#     mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

#     # Annotate on a copy of the original image
#     source_image_annotated = box_annotator.annotate(
#         scene=image_bgr.copy(),
#         detections=detections
#     )
#     segmented_image_annotated = mask_annotator.annotate(
#         scene=image_bgr.copy(),
#         detections=detections
#     )

#     # Save annotated results
#     Image.fromarray(source_image_annotated).save("results/segmentation/source_image.jpg")
#     Image.fromarray(segmented_image_annotated).save("results/segmentation/segmented_image.jpg")

#     # Save each mask individually
#     for i, mask_3c in enumerate(mask_list):
#         # mask_3c has shape (3, H, W); take channel 0 and scale to [0..255]
#         single_channel = (mask_3c[0].astype(np.uint8) * 255)
#         mask_pil = Image.fromarray(single_channel)
#         mask_pil.save(f"results/segmentation/mask_{i+1}_bgs.jpg")

#     return True
