import torch
from torchvision.ops import misc as misc_nn_ops
import backbone_resnet
from collections import OrderedDict
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict


class FeaturePyramidNetwork(nn.Module):

    def __init__(self,
                 in_channels_list,
                 out_channels,
                 extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks


    def get_result_from_inner_blocks(self, x, idx):
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1

        return out


    def get_result_from_layer_blocks(self, x, idx):
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1

        return out


    def forward(self, x : Dict[str, Tensor]):

        names = list(x.keys()) # layer names
        x = list(x.values()) # feature maps

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x)-2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:] # [batch size, channel, width, height]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(nn.Module):

    def forward(self, x, y, names):
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))

        return x, names


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
    backbone = backbone_resnet.__dict__[backbone_name](
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

    # FPN example
    from backbone_resnet import resnet50

    m = FeaturePyramidNetwork([10, 20, 30], 5)
    x = OrderedDict()
    x['feat0'] = torch.rand(1, 10, 64, 64)
    x['feat2'] = torch.rand(1, 20, 16, 16)
    x['feat3'] = torch.rand(1, 30, 8, 8)
    output = m(x)
    print([(k, v.shape) for k, v in output.items()])
    

    # FPN with Resnet example
    m = resnet50(pretrained=True)
    new_m = torchvision.models._utils.IntermediateLayerGetter(m,
                                                              {'layer1': 'feat1', 'layer3': 'feat2'})
    out = new_m(torch.rand(1, 3, 224, 224))
    print([(k, v.shape) for k, v in out.items()])

    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    x = torch.rand(1, 3, 64, 64)
    output = backbone(x)
    print([(k, v.shape) for k, v in output.items()])
