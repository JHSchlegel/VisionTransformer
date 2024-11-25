import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(
        self, input_channels=3, num_classes=10, initial_filters=16, block_configs=None
    ):
        """
        Initialize the ResNet with configurable parameters.

        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_classes (int): Number of output classes for classification.
            initial_filters (int): Number of filters in the first convolutional layer.
            block_configs (list of tuples): Configuration for each residual block layer.
                Each tuple should contain (num_blocks, out_channels, stride).
        """
        super().__init__()

        # Default block configuration if none provided
        if block_configs is None:
            block_configs = [
                (2, 32, 1),
                (2, 64, 2),
                (2, 128, 2),
            ]  # (blocks, out_channels, stride)

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=initial_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
        )

        # Dynamically create residual layers
        layers = []
        in_channels = initial_filters
        for num_blocks, out_channels, stride in block_configs:
            layers.append(
                self._make_residual_layer(in_channels, out_channels, num_blocks, stride)
            )
            in_channels = out_channels  # Update in_channels for the next layer

        self.residual_layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling for flexibility
        self.fc = nn.Linear(in_channels, num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.residual_layers(x)
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
