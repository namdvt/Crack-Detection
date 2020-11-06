import torch
import torch.nn as nn
import timm


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for c in self.modules():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # backbone
        self.backbone = timm.create_model('resnext101_32x8d', pretrained=True)
        self.linear = nn.Sequential(
            Linear(1000, 512),
            Linear(512, 256),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)

        return x


if __name__ == '__main__':
    sample = torch.zeros((2, 3, 224, 224))
    sample2 = torch.zeros((2, 4, 202, 202))
    net = Model()

    out = net(sample, sample2)
    print()
