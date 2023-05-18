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
    CHECKPOINT_PATH = "/home/ioanna/Documents/Thesis/src/sam_weights/sam_vit_b_01ec64.pth"
    DEVICE = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_b"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    # image = encode_image(image)
    image_bgr = cv2.imread(image)
    im_array = np.asarray(image_bgr)
    im_width, im_height = im_array.shape[0], im_array.shape[1]

    print(im_width, im_height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor.set_image(image_rgb)
    
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

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

    masks2, scores2, logits2 = mask_predictor.predict(
        box=box2,
        multimask_output=True
    )
    box_annotator2 = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator2 = sv.MaskAnnotator(color=sv.Color.red())

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    detections2 = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks2),
        mask=masks2
    )
    detections2 = detections2[detections2.area == np.max(detections2.area)]

    source_image1 = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
    segmented_image1 = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    source_image2 = box_annotator2.annotate(scene=image_bgr.copy(), detections=detections2, skip_label=True)
    segmented_image2 = mask_annotator2.annotate(scene=image_bgr.copy(), detections=detections2)



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


    source_image_array = Image.fromarray(source_image1)
    source_image_array.save("/home/ioanna/Documents/Thesis/src/temp/source_image.jpg")

    seg_image_array = Image.fromarray(segmented_image1)
    seg_image_array.save("/home/ioanna/Documents/Thesis/src/temp/segmented_image.jpg")
    
    mask_array = Image.fromarray(masks[0])
    mask_array.save("/home/ioanna/Documents/Thesis/src/temp/mask11.jpg")

    mask_array = Image.fromarray(masks[1])
    mask_array.save("/home/ioanna/Documents/Thesis/src/temp/mask12.jpg")

    mask_array = Image.fromarray(masks[2])
    mask_array.save("/home/ioanna/Documents/Thesis/src/temp/mask13.jpg")

    source_image_array = Image.fromarray(source_image2)
    source_image_array.save("/home/ioanna/Documents/Thesis/src/temp/source_image2.jpg")

    seg_image_array = Image.fromarray(segmented_image2)
    seg_image_array.save("/home/ioanna/Documents/Thesis/src/temp/segmented_image2.jpg")
    
    mask_array = Image.fromarray(masks2[0])
    mask_array.save("/home/ioanna/Documents/Thesis/src/temp/mask21.jpg")

    mask_array = Image.fromarray(masks2[1])
    mask_array.save("/home/ioanna/Documents/Thesis/src/temp/mask22.jpg")

    mask_array = Image.fromarray(masks2[2])
    mask_array.save("/home/ioanna/Documents/Thesis/src/temp/mask23.jpg")


    return True

bbox = {'x': 9, 'y': 40, 'width': 330, 'height': 412}
# bbox = {'x': 15, 'y': 16, 'width': 165, 'height': 237}

# # encode_image('moving.jpg')
result = seg_anything('moving.jpg', bbox)
