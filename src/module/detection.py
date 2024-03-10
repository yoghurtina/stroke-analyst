import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
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
    CHECKPOINT_PATH = "src/sam_weights/vit_huge.pth"
    # DEVICE = torch.device('cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    # sam.to(device=DEVICE)
    CHECKPOINT_PATH = "src/sam_weights/stroke_huge.pth"

    mask_predictor = SamPredictor(sam)
    image_bgr = cv2.imread(image)
    im_array = np.asarray(image_bgr)

    uploaded = Image.open(image)

    im_width, im_height = im_array.shape[0], im_array.shape[1]

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

        box2 = np.array([
            bbox['x'] + bbox['width'],
            bbox['y'] ,
            im_width, 
            bbox['y'] + bbox['height']
        ])
        print(box)
        print(box2)

        masks_hem1, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX) # INDEX # CLASS #TRACK
        
        masks_hem2, scores2, logits2 = mask_predictor.predict(
            box=box2,
            multimask_output=True
        )
        box_annotator2 = sv.BoxAnnotator(color=sv.Color.red())
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

        source_image_hem1 = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
        segmented_image_hem1 = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        source_image_hem2 = box_annotator2.annotate(scene=image_bgr.copy(), detections=detections2, skip_label=True)
        segmented_image_hem2 = mask_annotator2.annotate(scene=image_bgr.copy(), detections=detections2)

        # sv.plot_images_grid(
        # images=[source_image1, segmented_image1],
        # grid_size=(1, 2),
        # titles=['source image', 'segmented image']
        # )

        # sv.plot_images_grid(
        #     images=masks,
        #     grid_size=(1, 4),
        #     size=(16, 4)
        # )

        # sv.plot_images_grid(
        # images=[source_image2, segmented_image2],
        # grid_size=(1, 2),
        # titles=['source image', 'segmented image']
        # )

        # sv.plot_images_grid(
        #     images=masks2,
        #     grid_size=(1, 4),
        #     size=(16, 4)
        # )

    source_image_array = Image.fromarray(source_image_hem1)
    source_image_array.save("src/temp/detection/source_image_hem1.jpg")

    seg_image_array = Image.fromarray(segmented_image_hem1)
    seg_image_array.save("src/temp/detection/segmented_image_hem1.jpg")
    
    mask_array = Image.fromarray(masks_hem1[0])
    mask_array.save("src/temp/detection/mask1_hem1.jpg")

    mask_array = Image.fromarray(masks_hem1[1])
    mask_array.save("src/temp/detection/mask2_hem1.jpg")

    mask_array = Image.fromarray(masks_hem1[2])
    mask_array.save("src/temp/detection/mask3_hem1.jpg")

    source_image_array = Image.fromarray(source_image_hem2)
    source_image_array.save("src/temp/detection/source_image_hem2.jpg")

    seg_image_array = Image.fromarray(segmented_image_hem2)
    seg_image_array.save("src/temp/detection/segmented_image_hem2.jpg")
    
    mask_array = Image.fromarray(masks_hem2[0])
    mask_array.save("src/temp/detection/mask1_hem2.jpg")

    mask_array = Image.fromarray(masks_hem2[1])
    mask_array.save("src/temp/detection/mask2_hem2.jpg")

    mask_array = Image.fromarray(masks_hem2[2])
    mask_array.save("src/temp/detection/mask3_hem2.jpg")


    return True

# bbox = {'x': 9, 'y': 40, 'width': 330, 'height': 412}
# # bbox = {'x': 15, 'y': 16, 'width': 165, 'height': 237}

# # # encode_image('moving.jpg')
# result = seg_anything('moving.jpg', bbox)



def seg_anything_bgs(image, bbox):
    CHECKPOINT_PATH = "src/sam_weights/sam_vit_b_01ec64.pth"
    # DEVICE = torch.device('cpu')
    MODEL_TYPE = "vit_b"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    # sam.to(device=DEVICE)
    CHECKPOINT_PATH = "src/sam_weights/huge_I1065_B8.pth"

    mask_predictor = SamPredictor(sam)
    # image = encode_image(image)
    image_bgr = cv2.imread(image)
    im_array = np.asarray(image_bgr)

    uploaded = Image.open(image)

    im_width, im_height = im_array.shape[0], im_array.shape[1]

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

        box2 = np.array([
            bbox['x'] + bbox['width'],
            bbox['y'] ,
            im_width, 
            bbox['y'] + bbox['height']
        ])
        print(box)
        print(box2)

        masks_bgs, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX) # INDEX # CLASS #TRACK      

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks_bgs),
            mask=masks_bgs
        )
        detections = detections[detections.area == np.max(detections.area)]

        source_image_hem1 = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
        segmented_image_hem1 = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    source_image_array = Image.fromarray(source_image_hem1)
    source_image_array.save("src/temp/segmentation/source_image.jpg")

    seg_image_array = Image.fromarray(segmented_image_hem1)
    seg_image_array.save("src/temp/segmentation/segmented_image.jpg")
    
    mask_array = Image.fromarray(masks_bgs[0])
    mask_array.save("src/temp/segmentation/mask_bgs.jpg")

    mask_array = Image.fromarray(masks_bgs[1])
    mask_array.save("src/temp/segmentation/mask2_bgs.jpg")

    return True
