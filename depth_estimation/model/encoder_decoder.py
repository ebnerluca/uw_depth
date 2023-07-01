import torch.nn as nn
from torchvision.models import mobilenet_v2

from .layers import CombinedUpsample


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        # encoder is based on MobileNetV2
        self.original_model = mobilenet_v2(pretrained=True)

    def forward(self, x):

        features = []

        # extract features after all layers mobilenetv2 model and concatenate in a list
        # for later usage in skip connections
        features.append(x)
        for submodule_id, submodule in self.original_model.features._modules.items():
            features.append(submodule(features[-1]))

        return features


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        out_channels=48,
        prior_channels=0,
        decoder_width=0.6,
        single_channel_output=False,
    ) -> None:
        super(Decoder, self).__init__()

        decoder_channels = int(in_channels * decoder_width)

        # 1x1 convolution to reduce/expand channels
        self.conv1x1 = nn.Conv2d(
            in_channels, decoder_channels, kernel_size=1, stride=1, padding=1
        )

        # upsampling layers
        # in_channels equals current channels + concat channels + prior channels
        self.up0 = CombinedUpsample(
            decoder_channels // 1 + 320 + prior_channels, decoder_channels // 2
        )
        self.up1 = CombinedUpsample(
            decoder_channels // 2 + 160 + prior_channels, decoder_channels // 2
        )
        self.up2 = CombinedUpsample(
            decoder_channels // 2 + 64 + prior_channels, decoder_channels // 4
        )
        self.up3 = CombinedUpsample(
            decoder_channels // 4 + 32 + prior_channels, decoder_channels // 8
        )
        self.up4 = CombinedUpsample(
            decoder_channels // 8 + 24 + prior_channels, decoder_channels // 8
        )
        self.up5 = CombinedUpsample(
            decoder_channels // 8 + 16 + prior_channels, out_channels
        )

        # optional: 3x3 convolution, 1 channel output
        # for estimating depth directly from encoder-decoder
        self.single_channel_output = single_channel_output
        if self.single_channel_output:
            self.conv3x3 = nn.Conv2d(
                decoder_channels // 16 - prior_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, features, depth_prior):

        # use subset of intermediate features as skip connections
        skip0 = features[2]  # size 16 x 240 x 320
        skip1 = features[4]  # size 24 x 120 x 160
        skip2 = features[6]  # size 32 x 60 x 80
        skip3 = features[9]  # size 64 x 30 x 40
        skip4 = features[15]  # size 160 x 15 x 20
        skip5 = features[18]  # size 320 x 15 x 20
        out = features[19]  # size 1280 x 15 x 20

        # convolve input to match decoder channnels c_in -> c
        out = self.conv1x1(out)  # size c x 15 x 20

        # upsample together with skip connections and depth prior
        out = self.up0(out, skip5, depth_prior)  # size c//2 x 15 x 20
        out = self.up1(out, skip4, depth_prior)  # size c//2 x 15 x 20
        out = self.up2(out, skip3, depth_prior)  # size c//4 x 30 x 40
        out = self.up3(out, skip2, depth_prior)  # size c//8 x 30 x 40
        out = self.up4(out, skip1, depth_prior)  # size c//8 x 120 x 160
        out = self.up5(out, skip0, depth_prior)  # size c_out x 240 x 320

        # optional: final convolution to achieve single channel
        if self.single_channel_output:
            out = self.conv3x3(out)  # size 1 x 240 x 320

        return out
