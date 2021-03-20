import torch
import torchvision

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align

import _utils as det_utils

from typing import Optional, List, Dict, Tuple


def maskrcnn_inference(x, labels):
    
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob

class RoIHeads(nn.Module):

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,

                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,

                 score_thresh,
                 nms_thresh,
                 detections_per_img,

                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None
                 ):

        super(RoIHeads, self).__init__()

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor


    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True


    def postprocess_detections(self,
                               class_logits,   
                               box_regression,  
                               proposals,      
                               image_shapes   
                               ):

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,   
                proposals,     
                image_shapes,  
                targets=None   
                ):

        labels = None
        regression_targets = None
        matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)

        for i in range(num_images):

            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}

            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

            losses.update(loss_mask)

        return result, losses