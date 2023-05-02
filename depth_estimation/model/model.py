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

    def __init__(self, n_bins=128) -> None:
        super(UDFNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()  # output N x 46 x 240 x 320
        self.mViT = mViT(
            in_channels=48,  # decoder output plus sparse prior parametrization
            embedding_dim=48,
            patch_size=16,
            num_heads=4,
            num_query_kernels=48,
            n_bins=n_bins,
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

        self.n_bins = n_bins

    def forward(self, rgb, depth_prior):
        """Input:
        - rgb: RGB input image, Nx3x480x640
        - depth_prior: Parametrization of sparse prior guidance signal, Nx2x240x320"""

        # encode
        encoder_out = self.encoder(rgb)

        # decode
        decoder_out = self.decoder(encoder_out)

        # concat prior parametrization
        mvit_in = torch.cat((decoder_out, depth_prior), dim=1)

        # normed bin widths, range attention maps
        bin_widths_normed, range_attention_maps = self.mViT(mvit_in)

        # bin centers in [0,1]
        # print(f"bin_widths normed: {bin_widths_normed}")
        bin_edges_normed = torch.cumsum(bin_widths_normed, dim=1)
        bin_edges_normed = functional.pad(
            bin_edges_normed, (1, 0), value=0.0
        )  # add edge at zero
        # print(f"bin_edges normed: {bin_edges_normed}")
        bin_centers_normed = 0.5 * (bin_edges_normed[:, :-1] + bin_edges_normed[:, 1:])
        # print(f"bin centers normed: {bin_centers_normed}")

        # depth classification scores
        depth_scores = self.conv_out(range_attention_maps)
        # print(f"depth scores: {depth_scores.shape}")

        # linear combination
        prediction = torch.sum(
            depth_scores
            * bin_centers_normed.view(bin_centers_normed.size(0), self.n_bins, 1, 1),
            dim=1,
            keepdim=True,
        )
        # print(f"prediction: {prediction.shape}")

        return prediction


def test_simple():

    print("Testing SimpleEncoderDecoder with random input ...")

    # instantiate model
    model = SimpleEncoderDecoder(debug=True)

    # generate random input
    random_batch = torch.rand(2, 3, 480, 640)

    # inference
    out = model(random_batch)


def test_udfnet():

    print("Testing UDFNet with random input ...")

    # instantiate model
    udfnet = UDFNet(n_bins=100)

    # generate random input
    random_batch = torch.rand(4, 3, 480, 640)

    # inference
    out = udfnet(random_batch)

    print("Ok")


# to run this, use "python -m depth_estimation.model.model"
# otherwise the imports do not work as intended
# check https://stackoverflow.com/questions/72852/how-can-i-do-relative-imports-in-python
if __name__ == "__main__":
    # test_simple()
    test_udfnet()
