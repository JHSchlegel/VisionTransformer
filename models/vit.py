"""
This module contains the implementation of a Vision Transformer (ViT) model with
optional support for returning attention maps.
"""

# --------------------------------------------------------------------------- #
#                           Packages and Presets                              #
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#                             Vision Transformer                              #
# --------------------------------------------------------------------------- #
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 40,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # linear projection of flattened patches:
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token and positional embeding:
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # num_patches + 1 because of the CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_channels, image_height, image_width)
        # project and flatten patches:
        x = self.projection(
            x
        )  # (batch_size, embedding_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embedding_dim)

        # add CLS token:
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / self.head_dim**0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(
        self, x: torch.Tensor, return_attention_map: bool = False
    ) -> torch.Tensor:
        batch_size, num_patches, embed_dim = x.shape

        # Generate query, key, value tensors:
        qkv = self.qkv(x).reshape(
            batch_size, num_patches, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, num_heads, num_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute attention scores:
        # if we want to return the attention map, we need to return the attention weights
        # prior to the dropout and the subsequent multiplication with the value tensor
        # Hence, we won't use the more efficient scaled_dot_product_attention function
        if return_attention_map:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn_weights = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn_weights)
            attn = attn @ v

            # apply attention to value:
            x = attn.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
            x = self.proj(x)
            x = self.proj_dropout(x)
            return x, attn_weights
        else:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
            x = attn.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
            x = self.proj(x)
            x = self.proj_dropout(x)
            return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head attention:
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # MLP block:
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention_map: bool = False
    ) -> torch.Tensor:
        # Multi-head attention with residual connection:
        identity = x
        if return_attention_map:
            x_attn, attn_map = self.attn(self.norm1(x), return_attention_map)
            x = x_attn + identity
            x = x + self.mlp(self.norm2(x))
            return x, attn_map
        else:
            x = self.attn(self.norm1(x)) + identity
            x = x + self.mlp(self.norm2(x))
            return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Vision Transformer

        Args:
            image_size (int, optional): Size of the input image. Defaults to 32.
            patch_size (int, optional): Size of the patch. Defaults to 4.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of classes. Defaults to 10.
            embed_dim (int, optional): Dimension of the embedding. Defaults to 512.
            depth (int, optional): Number of transformer blocks. Defaults to 5.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            mlp_ratio (int, optional): MLP hidden dimension ratio. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Transformer Encoder
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention_map: bool = False
    ) -> torch.Tensor:
        """Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor.
            return_attention_map (bool, optional): Whether to return attention maps. Defaults to False.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]: Model output or output and attention maps.
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Transformer blocks
        if return_attention_map:
            attn_weights = []
            for block in self.transformer:
                x, attn_map = block(x, return_attention_map)
                attn_weights.append(attn_map)
        else:
            for block in self.transformer:
                x = block(x)

        # Classification from [CLS] token
        x = self.norm(x)
        x = x[:, 0]  # Take only the CLS token
        x = self.head(x)

        if return_attention_map:
            return x, attn_weights

        return x
