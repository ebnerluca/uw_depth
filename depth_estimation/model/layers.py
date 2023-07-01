import torch
import torch.nn as nn
import torch.nn.functional as functional


class CombinedUpsample(nn.Sequential):
    """Upsample an input x by interpolation, concatenate with a skip_input and additional_input
    followed by two convolutional layers. Additional input (can be None) is interpolated to match
    resolution of skip_input and also concatenated."""

    def __init__(self, in_channels, out_channels):
        super(CombinedUpsample, self).__init__()

        # convolutional layer A
        self.convA = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # leaky relu A for nonlinearity
        self.leakyreluA = nn.LeakyReLU(0.2)

        # convolutional layer B
        self.convB = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # leaky relu B for nonlinearity
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, skip_input, additional_input=None):

        # upscale x such that it matches shape of skip connection
        x_interpolated = functional.interpolate(
            x,
            size=[skip_input.size(2), skip_input.size(3)],
            mode="bilinear",
            align_corners=True,
        )

        # depth wise concatenate skip input
        out = torch.cat([x_interpolated, skip_input], dim=1)

        # additional input
        if additional_input is not None:
            # scale such that it matches target shape
            additional_input_interpolated = functional.interpolate(
                additional_input,
                size=[skip_input.size(2), skip_input.size(3)],
                mode="bilinear",
                align_corners=True,
            )

            # depth wise concatenate additional input
            out = torch.cat([out, additional_input_interpolated], dim=1)

        # convolutional layers and activation
        out = self.convA(out)
        out = self.leakyreluA(out)
        out = self.convB(out)
        out = self.leakyreluB(out)

        return out


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=16, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()

        # convolution to prepare features for patch embeddings
        self.embedding_convPxP = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # create 500 positional encodings as unique vectors with length embedding_dim
        self.positional_encodings = nn.Parameter(
            torch.rand(embedding_dim, 500), requires_grad=True
        )

        # transformer layer used by encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=1024,
        )

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=4,
        )

    def forward(self, x):

        # patch embeddings of length embedding_dim, size batch x dim x n_patches
        patch_embeddings = self.embedding_convPxP(x).flatten(2)

        # add positional encodings batchwise
        patch_embeddings += self.positional_encodings[
            :, : patch_embeddings.size(2)
        ].unsqueeze(
            0
        )  # unsqueeze at dim 0 to add batch dimension

        # encode patch embeddings with transformer
        patch_embeddings = patch_embeddings.permute(
            2, 0, 1
        )  # transformer expects n_patches x emb_dim x n_batches
        out = self.transformer_encoder(
            patch_embeddings
        )  # output is n_patches x emb_dim x n_batches

        return out


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, nk = K.size()
        assert (
            c == nk
        ), "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        out = torch.matmul(
            x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1)
        )  # .shape = n, hw, cout
        return out.permute(0, 2, 1).view(n, cout, h, w)
