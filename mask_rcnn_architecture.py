from collections import OrderedDict
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from backbone_resnet import resnet101
from neck_fpn import fpn
from anchor_generator import AnchorGenerator
from rpn import RPNHead, RPN
from roi_heads import RoiHeads
from transform import GeneralizedRCNNTransform

class FasterRCNNHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.box_layer = nn.Linear(in_channels, num_classes * 4)
        self.score_layer = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)
        bbox_deltas = self.box_layer(x)
        scores = self.score_layer(x)

        return scores, bbox_deltas

class FCHead(nn.Sequential):
    def __init__(self, in_channels, representation_size):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs

class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))


class MaskRCNN:
    def __init__(self,
                 backbone,
                 num_classes=None,
                 # transform parameters
                 min_size=800,
                 max_size=1333,
                 image_mean=None,
                 image_std=None,

                 anchor_sizes=(32, 64, 128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0),

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

        out_channels = 256

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                               mask_dim_reduced, num_classes)

        self.num_classes = 91
        self.backbone = backbone
        self.transform = GeneralizedRCNNTransform(800, 1333, None, None)

        anchor_sizes = [[anchor] for anchor in anchor_sizes]
        aspect_ratios = [aspect_ratios] * len(anchor_sizes)

        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n = {'testing': rpn_pre_nms_top_n_test}
        rpn_post_nms_top_n = {'testing': rpn_post_nms_top_n_test}

        self.rpn = RPN(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        box_head = FCHead(
            out_channels * resolution ** 2,
            representation_size
        )

        representation_size = 1024
        box_predictor = FasterRCNNHead(
            representation_size,
            self.num_classes
        )

        self.roi_heads = RoiHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor)

    def forward(self, images):

        images, image_sizes, targets = self.transform(images)

        outputs = self.backbone(images)
        outputs = fpn(outputs)
        outputs = {str(i): o for i, o in enumerate(outputs)}

        proposals = self.rpn(images, image_sizes, outputs)
        detections = self.roi_heads(outputs, proposals, image_sizes)

        return detections