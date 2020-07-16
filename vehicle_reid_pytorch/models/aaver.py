import copy
from vehicle_reid_pytorch.models import Baseline
from torch import nn
import torch
from vehicle_reid_pytorch.models.baseline import weights_init_classifier, weights_init_kaiming


class AAVER(Baseline):
    def __init__(self, with_kp=True, with_mask=False, *args, **kwargs):
        """AAVER

        Arguments:
            Baseline {[type]} -- [description]
        """
        super(AAVER, self).__init__(*args, **kwargs)
        conv1_input_ch = 3
        self.with_kp = with_kp
        self.with_mask = with_mask
        if self.with_kp:
            conv1_input_ch += 1
        if self.with_mask:
            conv1_input_ch += 4
        old_conv1 = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            conv1_input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.conv1.weight[:, :3, :, :].data.copy_(old_conv1.weight.data)

    def forward(self, image, **kwargs):
        if self.with_kp:
            kp_heatmap = kwargs["kp_heatmap"]
            image = torch.cat([image, kp_heatmap.unsqueeze(1)], dim=1)
        if self.with_mask:
            mask = kwargs["mask"][:, 1:5, :, :]
            image = torch.cat([image, mask], dim=1)
        return super(AAVER, self).forward(image, **kwargs)


if __name__ == "__main__":
    model = AAVER(333, 1, '/data/models/resnet50.pth',
                  'bnneck', 'after', 'resnet50', 'imagenet')

