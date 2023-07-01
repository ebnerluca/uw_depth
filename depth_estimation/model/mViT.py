import torch
import torch.nn as nn

from .layers import PatchTransformerEncoder, PixelWiseDotProduct


class mViT(nn.Module):
    """
    - in_channels: number of input channels per pixel
    - embedding_dimension: dimension of patch embeddings for transformer input
    - patch_size: size of patch used for each embedding is patch_size x patch_size
    - num_heads: number of parallel heads for attention
    - num_query_kernels: number of output kernels used to compute range attention maps
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
            nn.Linear(256, n_bins + 1),  # n bins plus max depth
            # nn.ReLU(),
            nn.Softplus(),  # replace ReLU because it causes dead neurons
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

        # kernels for attention maps, size N x NK x E
        attention_kernels = out[1 : self.num_query_kernels + 1, ...].permute(1, 0, 2)

        # estimating max depth and normed bin_widths
        eps = 0.1
        mlp_out = self.mlp(bins_head) + eps
        max_depth = mlp_out[:, 0].unsqueeze(1)
        bin_widths_normed = mlp_out[:, 1:]
        bin_widths_normed = bin_widths_normed / bin_widths_normed.sum(
            dim=1, keepdim=True
        )

        # range attention maps, size N x NK x h x w
        range_attention_maps = self.dot_product_layer(x, attention_kernels)

        return max_depth, bin_widths_normed, range_attention_maps
