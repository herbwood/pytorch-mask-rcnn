import numpy as np
import random
import torch
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
from mask_rcnn import maskrcnn_resnet50_fpn
import os

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def final_prediction(image, model, threshold=0.965):
    with torch.no_grad():
        outputs = model(image) 

    # list(dict_keys(['boxes', 'labels', 'scores', 'masks']))
    # boxes : coordinates of detected boxes
    # labels : class label index of detected boxes
    # scores : confidence score of detectd boxes
    # masks : #detected x masks
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    
    # only masks upper threshold left
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[:thresholded_preds_count]
    
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in outputs[0]['boxes'].detach().cpu()]
    boxes = boxes[:thresholded_preds_count]
    
    labels = [coco_names[i] for i in outputs[0]['labels']]

    return masks, boxes, labels


def draw_boxes_masks(image, masks, boxes, labels):

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        
        # colarize
        COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color

        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.addWeighted(image, 1, segmentation_map, 0.6, 0, image)

        x_min, y_min = map(int, boxes[i][0])
        x_max, y_max = map(int, boxes[i][1])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
        cv2.putText(image, labels[i], (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)

    return image


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', required=True, 
    #                     help='path to the input data')
    # parser.add_argument('-sp', '--savepath', required=True, 
    #                     help='path to save data')
    # args = vars(parser.parse_args())

    # image_path = args['input']
    # save_path = args['savepath']
    # filename = os.path.split('/')[-1]

    model = maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    transform = transforms.Compose([transforms.ToTensor()])

    # python mask_rcnn_images.py --input ../input/image1.jpg

    args_input = "image/cats.jpg"

    image_path = args_input
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    image = transform(image)

    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = final_prediction(image, model)
    result = draw_boxes_masks(original_image, masks, boxes, labels)

    cv2.imshow('Segmented image', result)
    cv2.waitKey(0)
    save_path = "image/segmented_cats.jpg"
    cv2.imwrite(save_path, result)
