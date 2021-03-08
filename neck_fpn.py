import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, outputs):
        super(FPN, self).__init__()
        self.outputs = outputs

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # output = F.upsample(x, size=(H, W), mode='nearest') + y
        output = F.interpolate(x, size=(H, W), mode="nearest") + y
        return output

    def forward(self, x):
        f2, f3, f4, f5 = self.outputs
        print(f2.size, f3.size(), f4.size(), f5.size())
        p5 = self.toplayer(f5)
        p4 = self._upsample_add(p5, self.latlayer1(f4))
        p3 = self._upsample_add(p4, self.latlayer2(f3))
        p2 = self._upsample_add(p3, self.latlayer3(f2))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return [p2, p3, p4, p5]

def fpn(outputs):
    return FPN(outputs)