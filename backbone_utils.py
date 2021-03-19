from torch import nn
from feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
import resnet
from collections import OrderedDict
import torch
import torchvision


class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers):

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()

        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x

        return out


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers, in_channels_list,
                 out_channels, extra_blocks):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        return x


def resnet_fpn_backbone(backbone_name, pretrained,
                        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                        # trainable_layers = 3,
                        returned_layers=None,
                        extra_blocks=None
                        ):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer
    )

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    return BackboneWithFPN(backbone, return_layers, in_channels_list,
                           out_channels, extra_blocks=extra_blocks)


if __name__ == "__main__":
    from resnet import resnet50

    m = resnet50(pretrained=True)
    new_m = torchvision.models._utils.IntermediateLayerGetter(m,
                                                              {'layer1': 'feat1', 'layer3': 'feat2'})
    out = new_m(torch.rand(1, 3, 224, 224))
    print([(k, v.shape) for k, v in out.items()])

    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    x = torch.rand(1, 3, 64, 64)
    output = backbone(x)
    print([(k, v.shape) for k, v in output.items()])
