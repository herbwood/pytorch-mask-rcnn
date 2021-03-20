import torch
from torch.nn import functional as F
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops

import _utils as det_utils
from _utils import ImageList

from anchor_utils import AnchorGenerator


class RPNHead(nn.Module):


    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)


    def forward(self, x):

        logits = []
        bbox_reg = []

        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):

    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)

    return layer


def concat_box_prediction_layers(box_cls, box_regression):

    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression):

        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):


    def __init__(self,
                 anchor_generator,
                 head,
                 fg_iou_thresh, 
                 bg_iou_thresh,
                 batch_size_per_image, 
                 positive_fraction,
                 pre_nms_top_n, 
                 post_nms_top_n, 
                 nms_thresh, 
                 score_thresh=0.0):

        super(RegionProposalNetwork, self).__init__()

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3


    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']


    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']


    def _get_top_n_idx(self, objectness, num_anchors_per_level):

        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):

            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors

        return torch.cat(r, dim=1)


    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):

        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []

        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores


    def forward(self,
                images,   
                features,    
                targets=None  
                ):

        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}

        return boxes, losses
    

if __name__ == "__main__":

    # RPNHead test
    from backbone_utils import resnet_fpn_backbone

    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    images = torch.randn((1, 3, 64, 64))
    output = backbone(images)

    features = [v for _, v in output.items()]
    head = RPNHead(256, 9)
    logits, bbox_reg = head(features)

    for logit in logits:
        print(logit.shape)

    for bbox in bbox_reg:
        print(bbox.shape)
    
    
    # RegionProposalNetwork test

    # generate dummy data
    image_sizes_list = []
    image_sizes = [img.shape[-2:] for img in images]

    for image_size in image_sizes:
        image_sizes_list.append((image_size[0], image_size[1]))
    image_list = ImageList(images, image_sizes_list)


    # RPN parameters
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    head = RPNHead(256, 9)

    pre_nms_top_n = {'training' : 2000, 'testing' : 1000}
    post_nms_top_n = {'training' : 2000, 'testing' : 1000}

    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 256
    positive_fraction = 0.5
    nms_thresh = 0.7

    rpn = RegionProposalNetwork(anchor_generator=anchor_generator,
                                pre_nms_top_n=pre_nms_top_n,
                                post_nms_top_n=post_nms_top_n,
                                head=head,
                                fg_iou_thresh=fg_iou_thresh,
                                bg_iou_thresh=bg_iou_thresh,
                                batch_size_per_image=batch_size_per_image,
                                positive_fraction=positive_fraction,
                                nms_thresh=nms_thresh)

    features = backbone(images)
    boxes, _ = rpn(image_list, features)

    print(boxes[0].shape)    
