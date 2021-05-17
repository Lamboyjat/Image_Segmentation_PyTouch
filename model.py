import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# The first layer of the UNET Architecture taking input and performing two convolution
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        def forward(self, x):
            return self.conv(x)


# The second layer is trying to get the features at each down sampling and and perform Convolution on it again
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        ##The up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connection = []

        for down in self.downs:
            x = down(x)
            skip_connection.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx(x)]
            skip_connection = skip_connection[idx // 2]

            # if the input feature is a 161 x 161
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Otherwise keep the size feature
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    predict = model(x)
    print(predict.shape)
    print(x.shape)
    assert predict.shape == x.shape


if __name__ == "__main__":
    test()
