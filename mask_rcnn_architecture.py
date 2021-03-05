from collections import OrderedDict

from torch import nn

from torchvision.ops import MultiScaleRoIAlign

from utils import overwrite_eps
from utils import load_state_dict_from_url

# from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers

class MaskRCNN:
    def __init__(self,
                 backbone,
                 num_classes=None,

                 # transform parameters
                 min_size=800,
                 max_size=1333,
                 image_mean=None,
                 image_std=None,

                 # RPN parameters
                 rpn_anchor_generator=None,
                 rpn_head = None,
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256,
                 rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,

                 # Box paramters
                 box_roi_pool=None,
                 box_head = None,
                 box_predictor = None,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512,
                 box_positive_fraction=0.25,
                 bbox_reg_weights=None,

                 # Mask parameters
                 mask_roi_pool=None,
                 mask_head = None,
                 mask_predictor=None
                 ):




