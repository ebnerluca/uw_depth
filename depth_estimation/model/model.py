import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.linalg import lstsq
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

    def __init__(self, n_bins=128, true_scale_output=False) -> None:
        super(UDFNet, self).__init__()

        # encoder based on MobileNetV2
        self.encoder = Encoder()

        # decoder
        prior_channels = 2
        self.decoder = Decoder(
            in_channels=1280,
            out_channels=(48 - prior_channels),
            prior_channels=prior_channels,
        )  # output N x C x 240 x 320

        # mViT
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
        self.true_scale_output = true_scale_output

    def forward(self, rgb, prior_parametrization, prior_features=None):
        """Input:
        - rgb: RGB input image, Nx3x480x640
        - prior_parametrization: Parametrization of sparse prior guidance signal, Nx2x240x320
        - prior_features: list of features with their pixel position and depth value NxSx3, only used when outputting true scale ground_truth."""

        # encode
        encoder_out = self.encoder(rgb)

        # decode
        decoder_out = self.decoder(encoder_out, prior_parametrization)

        # concat prior parametrization
        mvit_in = torch.cat((decoder_out, prior_parametrization), dim=1)

        # normed bin widths, range attention maps
        bin_widths_normed, range_attention_maps = self.mViT(mvit_in)

        # bin centers in [0,1]
        bin_edges_normed = torch.cumsum(bin_widths_normed, dim=1)
        bin_edges_normed = functional.pad(
            bin_edges_normed, (1, 0), value=0.0
        )  # add edge at zero
        bin_centers_normed = 0.5 * (bin_edges_normed[:, :-1] + bin_edges_normed[:, 1:])

        # depth classification scores
        depth_scores = self.conv_out(range_attention_maps)

        # linear combination
        prediction = torch.sum(
            depth_scores
            * bin_centers_normed.view(bin_centers_normed.size(0), self.n_bins, 1, 1),
            dim=1,
            keepdim=True,
        )

        # recover true scale for each image in batch
        if self.true_scale_output:
            for i in range(prediction.size(0)):

                n_features = len(prior_features[i])

                # depth values of features
                feature_depth_values = prior_features[i, :, 2]

                # depth values of prediction at same location as in prior
                idcs_height = prior_features[i, :, 0].long()
                idcs_width = prior_features[i, :, 1].long()
                predicted_depth_values = prediction[
                    i, 0, idcs_height, idcs_width
                ].unsqueeze(1)

                # find least square solution for scale and offset
                # [prediction, 1]*[scale, offset] = true values
                # Ax = B
                A = torch.cat(
                    (
                        predicted_depth_values,
                        torch.ones(n_features, 1, device=predicted_depth_values.device),
                    ),
                    dim=1,
                )
                scale_and_offset = lstsq(A, feature_depth_values.unsqueeze(1)).solution

                # apply scale and offset to prediction
                prediction[i, ...] = (
                    prediction[i, ...] * scale_and_offset[0] + scale_and_offset[1]
                )

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
