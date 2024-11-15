import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer2 = self._make_residual_layer(16, 32, 2, 1)
        self.layer3 = self._make_residual_layer(32, 64, 2, 2)
        self.layer4 = self._make_residual_layer(64, 128, 2, 2)
        self.layer5 = self._make_residual_layer(128, 256, 2, 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(3200, 6)

    def forward(self, x):
        """Forward pass of network."""
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _make_residual_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)


# see: https://stackoverflow.com/questions/55688645/how-downsample-work-in-resnet-in-pytorch-code
# and https://discuss.pytorch.org/t/downsampling-at-resnet/39038
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out
