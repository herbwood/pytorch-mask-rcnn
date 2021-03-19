# import torch
# from torch import nn
# import torch.nn.functional as F

# from torchvision.ops import MultiScaleRoIAlign

# from _utils import overwrite_eps
# from _utils import load_state_dict_from_url

# from anchor_utils import AnchorGenerator
# # from generalized_rcnn import GeneralizedRCNN
# from rpn import RPNHead, RegionProposalNetwork
# from roi_heads import RoIHeads
# from transform import GeneralizedRCNNTransform
# from backbone_utils import resnet_fpn_backbone
# from collections import OrderedDict


# class FasterRCNN(nn.Module):


#     def __init__(self, backbone, num_classes=None,
#                  # transform parameters
#                  min_size=800, max_size=1333,
#                  image_mean=None, image_std=None,
#                  # RPN parameters
#                  rpn_anchor_generator=None, rpn_head=None,
#                  rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
#                  rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
#                  rpn_nms_thresh=0.7,
#                  rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
#                  rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
#                  rpn_score_thresh=0.0,
#                  # Box parameters
#                  box_roi_pool=None, box_head=None, box_predictor=None,
#                  box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
#                  box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
#                  box_batch_size_per_image=512, box_positive_fraction=0.25,
#                  bbox_reg_weights=None):

#         super(FasterRCNN, self).__init__()

#         if not hasattr(backbone, "out_channels"):
#             raise ValueError(
#                 "backbone should contain an attribute out_channels "
#                 "specifying the number of output channels (assumed to be the "
#                 "same for all the levels)")

#         assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
#         assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

#         if num_classes is not None:
#             if box_predictor is not None:
#                 raise ValueError("num_classes should be None when box_predictor is specified")
#         else:
#             if box_predictor is None:
#                 raise ValueError("num_classes should not be None when box_predictor "
#                                  "is not specified")
        

#         out_channels = backbone.out_channels

#         self.backbone = backbone

#         if rpn_anchor_generator is None:
#             anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#             aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#             rpn_anchor_generator = AnchorGenerator(
#                 anchor_sizes, aspect_ratios
#             )
#         if rpn_head is None:
#             rpn_head = RPNHead(
#                 out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
#             )

#         rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
#         rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

#         self.rpn = RegionProposalNetwork(
#             rpn_anchor_generator, rpn_head,
#             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
#             rpn_batch_size_per_image, rpn_positive_fraction,
#             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
#             score_thresh=rpn_score_thresh)

#         if box_roi_pool is None:
#             box_roi_pool = MultiScaleRoIAlign(
#                 featmap_names=['0', '1', '2', '3'],
#                 output_size=7,
#                 sampling_ratio=2)

#         if box_head is None:
#             resolution = box_roi_pool.output_size[0]
#             representation_size = 1024
#             box_head = TwoMLPHead(
#                 out_channels * resolution ** 2,
#                 representation_size)

#         if box_predictor is None:
#             representation_size = 1024
#             box_predictor = FastRCNNPredictor(
#                 representation_size,
#                 num_classes)

#         self.roi_heads = RoIHeads(
#             # Box
#             box_roi_pool, box_head, box_predictor,
#             box_fg_iou_thresh, box_bg_iou_thresh,
#             box_batch_size_per_image, box_positive_fraction,
#             bbox_reg_weights,
#             box_score_thresh, box_nms_thresh, box_detections_per_img)

#         if image_mean is None:
#             image_mean = [0.485, 0.456, 0.406]
#         if image_std is None:
#             image_std = [0.229, 0.224, 0.225]
#         self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        


#     def forward(self, images, targets=None):
    
#         original_image_sizes = []

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


# class TwoMLPHead(nn.Module):
#     """
#     Standard heads for FPN-based models
#     Args:
#         in_channels (int): number of input channels
#         representation_size (int): size of the intermediate representation
#     """

#     def __init__(self, in_channels, representation_size):
#         super(TwoMLPHead, self).__init__()

#         self.fc6 = nn.Linear(in_channels, representation_size)
#         self.fc7 = nn.Linear(representation_size, representation_size)

#     def forward(self, x):
#         x = x.flatten(start_dim=1)

#         x = F.relu(self.fc6(x))
#         x = F.relu(self.fc7(x))

#         return x


# class FastRCNNPredictor(nn.Module):
#     """
#     Standard classification + bounding box regression layers
#     for Fast R-CNN.
#     Args:
#         in_channels (int): number of input channels
#         num_classes (int): number of output classes (including background)
#     """

#     def __init__(self, in_channels, num_classes):
#         super(FastRCNNPredictor, self).__init__()
#         self.cls_score = nn.Linear(in_channels, num_classes)
#         self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

#     def forward(self, x):
#         if x.dim() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#         x = x.flatten(start_dim=1)
#         scores = self.cls_score(x)
#         bbox_deltas = self.bbox_pred(x)

#         return scores, bbox_deltas


# model_urls = {'fasterrcnn_resnet50_fpn_coco':
#                 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',}


# def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
#                             num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):

#     if pretrained:
#         # no need to download the backbone if pretrained is set
#         pretrained_backbone = False
#     backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
#     model = FasterRCNN(backbone, num_classes, **kwargs)
#     print(model.parameters())
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
#                                               progress=progress)
#         print(state_dict.keys())
#         model.load_state_dict(state_dict)
#         overwrite_eps(model, 0.0)
#     return model