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



    CHECKPOINT_PATH = "src/sam_weights/meta_base.pth"
    DEVICE = 'cpu'
    MODEL_TYPE = "vit_b"
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


# def seg_anything_bgs(image_path, bbox):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     CHECKPOINT_PATH = "src/sam_weights/bgs_base_cpu.pth"

#     # Assuming the model configuration and processor are compatible with your fine-tuned model
#     model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
#     processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

#     # Create an instance of the model architecture with the loaded configuration
#     my_model = SamModel(config=model_config)

#     # Load the model weights and ensure the model is on the correct device
#     my_model.load_state_dict(torch.load("src/sam_weights/bgs_base_cpu.pth", map_location=device))
#     my_model = my_model.to(device)

#     # Process the image and bounding box
#     image = Image.open(image_path).convert("RGB")
#     bbox_list = [[[
#         float(bbox['x']),
#         float(bbox['y']),
#         float(bbox['x'] + bbox['width']),
#         float(bbox['y'] + bbox['height'])
#     ]]]  

#     # Process the image with the bounding box
#     inputs = processor(images=image, input_boxes=bbox_list, return_tensors="pt").to(device)  

#     with torch.no_grad():
#         outputs = my_model(**inputs, multimask_output=False)

#     medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
#     # convert soft mask to hard mask
#     medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
#     medsam_seg = (medsam_seg_prob == 1.0).astype(np.uint8)

#     # fore thresholding, check the distribution of predicted probabilities
#     print("Max probability:", medsam_seg_prob.max())
#     print("Min probability:", medsam_seg_prob.min())
#     print("Mean probability:", medsam_seg_prob.mean())

#     mask_pil = Image.fromarray((medsam_seg * 255).astype(np.uint8))
#     mask_path = f"results/segmentation/mask_bgs.jpg"
#     mask_pil.save(mask_path)
#     mask_pil = mask_pil.convert("L")
#     mask_pil = mask_pil.resize(image.size, Image.Resampling.LANCZOS)

#     # Overlay mask onto the original image to create the segmented image
#     segmented = Image.new("RGBA", image.size, (0, 0, 0, 0))   
#     image_rgba = image.convert("RGBA") 
#     segmented.paste(image, (0, 0), mask_pil)
#     seg_image_path = f"results/segmentation/segmented_image.png"
#     segmented.save(seg_image_path)

#     # Save the original image as well
#     source_image_path = f"results/segmentation/source_image.jpg"
#     image.save(source_image_path)

#     # Draw the bounding box on the original image and save it
#     draw = ImageDraw.Draw(image)
#     draw.rectangle([(bbox['x'], bbox['y']), (bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])], outline="red", width=3)
#     image_with_box_path = f"results/segmentation/image_with_box.jpg"
#     image.save(image_with_box_path)
#     return True


def seg_anything_bgs(image, bbox):
    CHECKPOINT_PATH = "src/sam_weights/meta_base.pth"
    # DEVICE = torch.device('cpu')
    MODEL_TYPE = "vit_b"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    # sam.to(device=DEVICE)
    # CHECKPOINT_PATH = "src/sam_weights/huge_I1065_B8.pth"

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
        box_annotator = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.INDEX)
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX) # INDEX # CLASS #TRACK      

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks_bgs),
            mask=masks_bgs
        )
        detections = detections[detections.area == np.max(detections.area)]

        source_image_annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image_annotated = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    Image.fromarray(source_image_annotated).save("results/segmentation/source_image.jpg")
    Image.fromarray(segmented_image_annotated).save("results/segmentation/segmented_image.jpg")
    
    mask_array = Image.fromarray(masks_bgs[0])
    mask_array.save("results/segmentation/mask_bgs.jpg")

    mask_array = Image.fromarray(masks_bgs[1])
    mask_array.save("results/segmentation/mask2_bgs.jpg")

    return True