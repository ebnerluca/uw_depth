import torch
import torch.nn as nn
import torch.nn.functional as functional

from .encoder_decoder import Encoder, Decoder
from .mViT import mViT


class SimpleEncoderDecoder(nn.Module):
    def __init__(self, debug=False) -> None:
        super(SimpleEncoderDecoder, self).__init__()

        self.debug = debug

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded_features = self.encoder(x)
        if self.debug:
            for feature in encoded_features:
                print(f"Encoded features shape: {feature.shape}")

        decoder_out = self.decoder(encoded_features)
        if self.debug:
            print(f"decoded features shape: {decoder_out.shape}")

        return decoder_out


class UDFNet(nn.Module):
    """Underwater Depth Fusion Net"""

    def __init__(self, n_bins=128, max_depth=None) -> None:
        super(UDFNet, self).__init__()

        # encoder based on MobileNetV2
        self.encoder = Encoder()

        # decoder
        prior_channels = 2  # channels of prior parametrization
        self.decoder = Decoder(
            in_channels=1280,
            out_channels=(48 - prior_channels),
            prior_channels=prior_channels,
        )  # output N x C x 240 x 320

        # mViT
        self.mViT = mViT(
            in_channels=48,  # decoder output plus prior parametrization
            embedding_dim=48,
            patch_size=16,
            num_heads=4,
            num_query_kernels=48,
            n_bins=n_bins,
        )

        # regression for bin scores
        self.conv_out = nn.Sequential(
            nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

        # params
        self.n_bins = n_bins
        self.max_depth = max_depth

    def forward(self, rgb, prior_parametrization):
        """Input:
        - rgb: RGB input image, Nx3x480x640
        - prior_parametrization: Parametrization of sparse prior guidance signal, Nx2x240x320"""

        # encode
        encoder_out = self.encoder(rgb)

        # decode
        decoder_out = self.decoder(encoder_out, prior_parametrization)

        # concat prior parametrization
        mvit_in = torch.cat((decoder_out, prior_parametrization), dim=1)

        # normed bin widths, range attention maps
        pred_max_depth, bin_widths_normed, range_attention_maps = self.mViT(mvit_in)

        # bin edges
        bin_edges_normed = torch.cumsum(bin_widths_normed, dim=1)
        bin_edges_normed = functional.pad(
            bin_edges_normed, (1, 0), value=0.0
        )  # add edge at zero

        # scale bin edges
        if self.max_depth is None:
            bin_edges = bin_edges_normed * pred_max_depth
        else:
            bin_edges = bin_edges_normed * self.max_depth

        # bin centers
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # depth classification scores
        depth_scores = self.conv_out(range_attention_maps)

        # linear combination of centers and scores
        prediction = torch.sum(
            depth_scores * bin_centers.view(bin_centers.size(0), self.n_bins, 1, 1),
            dim=1,
            keepdim=True,
        )

        # return prediction and bin edges (for loss and visualization)
        return prediction, bin_edges


def test_simple():

    print("Testing SimpleEncoderDecoder with random input ...")

    # instantiate model
    model = SimpleEncoderDecoder(debug=True)

    # generate random input
    random_batch = torch.rand(2, 3, 480, 640)

    # inference
    out = model(random_batch)

    print("Ok")


def test_udfnet():

    print("Testing UDFNet with random input ...")

    # instantiate model
    udfnet = UDFNet(n_bins=100)

    # generate random input
    random_rgb = torch.rand(4, 3, 480, 640)
    random_prior = torch.rand(4, 2, 240, 320)

    # inference
    out = udfnet(random_rgb, random_prior)

    print("Ok")


# to run this, use "python -m depth_estimation.model.model"
# otherwise the imports do not work as intended
# check https://stackoverflow.com/questions/72852/how-can-i-do-relative-imports-in-python
if __name__ == "__main__":
    # test_simple()
    test_udfnet()
