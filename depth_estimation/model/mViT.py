import torch
import torch.nn as nn
from .layers import PatchTransformerEncoder, PixelWiseDotProduct


class mViT(nn.Module):
    """
    Args
        in_channels: number of input channels per pixel
        embedding_dimension: dimension of patch embeddings for transformer input
        patch_size: size of patch used for each embedding is patch_size x patch_size
        num_heads: number of parallel heads for attention
        num_query_kernels: number of output kernels used to compute range attention maps
    """

    def __init__(
        self,
        in_channels,
        embedding_dim=48,
        patch_size=16,
        num_heads=4,
        num_query_kernels=48,
        n_bins=128,
    ) -> None:
        super(mViT, self).__init__()

        # patch transformer
        self.patch_transformer_encoder = PatchTransformerEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,  # E (embeddings dimension)
            patch_size=patch_size,
            num_heads=num_heads,
        )
        self.num_query_kernels = num_query_kernels  # NK (num kernels)

        # multi layer perceptron for bin widths
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_bins),
        )
        self.n_bins = n_bins

        # dot product layer for computing range attention map from decoder output and learned 1x1 kernels
        self.dot_product_layer = PixelWiseDotProduct()

    def forward(self, x):

        out = self.patch_transformer_encoder(
            x.clone()  # TODO test if it works without clone?
        )  # S x E x N (S: num patches aka. sequence length)

        # regression head for adaptive bins, size N x E
        bins_head = out[0, ...]
        # print(f"bins_head: {bins_head.shape}")

        # kernels for attention maps, size N x NK x E
        attention_kernels = out[1 : self.num_query_kernels + 1, ...].permute(1, 0, 2)
        # print(f"attention_kernel: {attention_kernels.shape}")

        # bin widths
        bin_widths_normed = self.mlp(bins_head)
        eps = 0.1  # numerical stability
        # eps = 1.0 / self.n_bins  # bias: assume all bins equal length
        bin_widths_normed = torch.relu(bin_widths_normed) + eps  # non negative
        bin_widths_normed /= bin_widths_normed.sum(dim=1, keepdim=True)  # unit length
        # print(f"bin_widths_normed: {bin_widths_normed.shape}")

        # range attention maps, size N x NK x h x w
        range_attention_maps = self.dot_product_layer(x, attention_kernels)
        # print(f"range_attention_maps: {range_attention_maps.shape}")

        return bin_widths_normed, range_attention_maps
