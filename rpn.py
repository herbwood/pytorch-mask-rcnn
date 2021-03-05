import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchvision
from torchvision.ops.boxes import batched_nms, box_iou, clip_boxes_to_image, remove_small_boxes

from typing import List, Optional, Dict, Tuple
from utils import ImageList, BoxCoder, Matcher, BalancedPositiveNegativeSampler

# classification, bbox regression을 수행하는 네트워크
class RPNHead(nn.Module):

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # cls는 anchor 수만큼, bbox는 anchor x 4만큼의 channel 생성
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors*4, kernel_size=1, stride=1)

        # 가중치 초기화
        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    # x : 하나의 이미지에 대한 4개의 feature pyramid
    def forward(self, x):
        logits = []
        bbox_reg = []

        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg

# prediction의 크기 reshape
def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2) # (N, H, W, -1, C)
    layer = layer.reshape(N, -1, C) # (N, HxWx(-1), C)

    return layer

# logits, bbox_reg를 입력받아 box prediction 출력
def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level) # (N, HxWx(-1), C)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level) # # (N, HxWx(-1), C)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2) #
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4) # 모든 cell별 bbox offset

    return box_cls, box_regression

class RPN(nn.Module):
    def __init__(self,
                 anchor_generator,
                 head,
                 fg_iou_thresh,
                 bg_iou_thresh,
                 batch_size_per_image,
                 positive_fraction,
                 pre_nms_top_n : Dict[str, int],
                 post_nms_top_n : Dict[str, int],
                 nms_thresh,
                 score_thresh=0.0):

        super(RPN, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # IoU 계산
        self.box_similarity = box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3


    # pre NMS할 RoI의 수 반환(training, testing 시 다르게)
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']


    # post NMS할 RoI의 수 반환(training, testing 시 다르게)
    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']


    def _get_top_n_idx(self,
                       objectness,
                       num_anchors_per_level : List[int]) -> Tensor:
        r = []
        offset = 0

        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors

        return torch.cat(r, dim=1)


    def filter_proposals(self,
                         proposals,
                         objectness,
                         image_shapes,
                         num_anchors_per_level):

        # number of proposals
        num_images = proposals.shape[0]
        device = proposals.device

        # objectness score per proposals
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # [[0, 0, ..., 0], [1, 1, ..., 1], ..., [n, n, ..., n]]
        # pyramid level per proposals
        levels = [
            torch.full((n,), i, dtype=torch.int64, device=device)
            for i, n in enumerate(num_anchors_per_level)
        ]

        levels = torch.cat(levels, dim=0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # NMS 적용 전에 level별로 top-n box를 선택
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        # NMS 적용 전에 top-n의 objectness score, level, proposals
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []

        for boxes, scores, lv1, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = clip_boxes_to_image(boxes, img_shape)

            # 작은 box 제거
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # score가 작은 box 제거
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # level별로 NMS
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

             # top-n개의 prediction만 보존
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

    def forward(self,
                images,
                sizes,
                features,
                targets):

        features = list(features.values())

        # pre_bbox_deltas : not offset, regressors
        objectness, pre_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, sizes, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pre_bbox_deltas)

        proposals = self.box_coder.decode(pre_bbox_deltas.detach(), anchors)

        # bbox location per images
        proposals = proposals.view(num_images, -1, 4)
        
        boxes, scores = self.filter_proposals(proposals, objectness, sizes, num_anchors_per_level)

        return boxes





