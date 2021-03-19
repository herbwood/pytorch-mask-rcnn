# from collections import OrderedDict
# import torch
# from torch import nn, Tensor
# import warnings
# from typing import Tuple, List, Dict, Optional, Union


# class GeneralizedRCNN(nn.Module):

#     def __init__(self, backbone, rpn, roi_heads, transform):
#         super(GeneralizedRCNN, self).__init__()
#         self.transform = transform
#         self.backbone = backbone
#         self.rpn = rpn
#         self.roi_heads = roi_heads
#         self._has_warned = False

#     def forward(self, images, targets=None):

#         original_image_sizes: List[Tuple[int, int]] = []

#         for img in images:
#             val = img.shape[-2:]
#             assert len(val) == 2
#             original_image_sizes.append((val[0], val[1]))

#         images, targets = self.transform(images, targets)

#         features = self.backbone(images.tensors)

#         if isinstance(features, torch.Tensor):
#             features = OrderedDict([('0', features)])

#         proposals, proposal_losses = self.rpn(images, features, targets)
#         detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
#         detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

#         return detections