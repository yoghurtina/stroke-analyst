import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from collections import defaultdict

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=lz7B4NDoJRxJ
# https://colab.research.google.com/drive/1SNzKH_W0cZ-UllFwxUjxi7Ow8qwPdlC1#scrollTo=Ug6MN1IcqEm-

CHECKPOINT_PATH = "src/sam_weights/sam_vit_b_01ec64.pth"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"

# extract the bounding box coordinates which will be used to feed into SAM as prompts.
bbox_coords = {}
for f in sorted(Path('drive/MyDrive/training_data/stroke_bounding_boxes').iterdir()): # bounding boxes masks
    k = f.stem[4:]
    im = cv2.imread(f.as_posix())
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(contours) > 1:
        x,y,w,h = cv2.boundingRect(contours[0])
        height, width, _ = im.shape
        bbox_coords[k] = np.array([x, y, x + w, y + h])

# extract the ground truth segmentation masks
ground_truth_masks = {}
for k in bbox_coords.keys():
    gt_grayscale = cv2.imread(f'drive/MyDrive/training_data/stroke_extracted_normalized/se_{k}.jpg', cv2.IMREAD_GRAYSCALE) # segmented piece from original gray photo   
    ground_truth_masks[k] = (gt_grayscale == 0)

sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam_model.to(DEVICE)
sam_model.train();


# Preprocess the images
# convert the input images into a format SAM's internal functions expect.

transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
    image = cv2.imread(f'drive/MyDrive/training_data/normalized/s{k}.jpg') # original image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=DEVICE)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size


# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(bbox_coords.keys())

# run fine-tuning

from statistics import mean
from torch.nn.functional import threshold, normalize

num_epochs = 100
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    # Just train on the first 20 examples
    for k in keys[:20]:
        input_image = transformed_data[k]['image'].to(DEVICE)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

    # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            
            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=DEVICE)
            box_torch = box_torch[None, :]
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(DEVICE)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(DEVICE)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    losses.append(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')