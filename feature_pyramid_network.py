from collections import OrderedDict
import torch.nn.functional as F
from torch import nn, Tensor
import torch
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


if __name__ == "__main__":
    m = FeaturePyramidNetwork([10, 20, 30], 5)
    x = OrderedDict()
    x['feat0'] = torch.rand(1, 10, 64, 64)
    x['feat2'] = torch.rand(1, 20, 16, 16)
    x['feat3'] = torch.rand(1, 30, 8, 8)
    output = m(x)
    print([(k, v.shape) for k, v in output.items()])