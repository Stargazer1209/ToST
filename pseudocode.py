import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class PreNorm(nn.Module):
    """Pre-normalization module"""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)

class TSSAttention(nn.Module):
    """Token Statistics Self Attention"""
    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape  # batch, tokens, dim
        
        # Generate Q/K/V
        w = self.to_qkv(x)
        w = rearrange(w, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Normalize
        w_normed = F.normalize(w, dim=-2)
        
        # Calculate statistics attention
        Pi = self.attend(torch.sum(w_normed ** 2, dim=-1) * self.temp)
        Pi = self.dropout(Pi)
        
        # Attention calculations
        dots = torch.matmul(
            (Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2),
            w ** 2
        )
        attn = 1. / (1 + dots)
        
        # Apply attention to values
        out = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, 
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: nn.Module = nn.GELU,
                 dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Transformer(nn.Module):
    """Transformer with TSS Attention"""
    def __init__(self, 
                 dim: int, 
                 depth: int, 
                 heads: int, 
                 dim_head: int = 64,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TSSAttention(
                    dim=dim,
                    num_heads=heads,
                    dim_head=dim_head,
                    dropout=dropout
                )),
                PreNorm(dim, MLP(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                    dropout=dropout
                ))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            # Attention with residual
            x = attn(x) + x
            # FFN with residual
            x = ff(x) + x
        return x