import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# l2 distance related

def prepare_l2_dist_squared_matmul(x, y):
    x_sum_squares = x.pow(2).sum(dim = -1, keepdim = True)
    y_sum_squares = y.pow(2).sum(dim = -1, keepdim = True)

    x = torch.cat((
        -2 * x,
        torch.ones_like(x_sum_squares),
        x_sum_squares
    ), dim = -1)

    y = torch.cat((
        y,
        y_sum_squares,
        torch.ones_like(y_sum_squares)
    ), dim = -1)

    return x, y

def l2_dist_squared(x, y):
    x, y = prepare_l2_dist_squared_matmul(x, y)
    return x @ y.t()

# l2 distance linear

class L2DistanceLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        squared = True,
        bias = False,
        negate = False
    ):
        super().__init__()

        self.negate = negate
        self.squared = squared

        self.weights = nn.Parameter(torch.randn(dim_out, dim))
        self.bias = nn.Parameter(torch.randn(dim_out)) if bias else None

    def forward(self, x):
        w = self.weights

        if self.squared:
            x = l2_dist_squared(x, w)
        else:
            x = torch.cdist(x, w)

        if self.negate:
            x = -x

        if exists(self.bias):
            x = x + self.bias

        return x

# class

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        l2_dist_proj_out = True
    ):
        super().__init__()
        dim_inner = heads * dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = L2DistanceLinear(dim, dim_inner * 3, squared = False)

        self.split_heads = Rearrange('b n (qkv h d) -> qkv (b h) n d', qkv = 3, h = heads)
        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)

        self.to_out = L2DistanceLinear(dim_inner, dim, squared = False) if l2_dist_proj_out else nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x):

        qkv = self.to_qkv(x)
        q, k, v = self.split_heads(qkv)

        q, k = prepare_l2_dist_squared_matmul(q, k)

        neg_l2_dist_sq = -einsum(q, k, 'b i d, b j d -> b i j')

        sim = neg_l2_dist_sq * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b i j, b j d -> b i d')

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        l2_dist_proj_out = True
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.proj_in = L2DistanceLinear(dim, dim_inner, negate = True, bias = True)
        self.proj_out = L2DistanceLinear(dim_inner, dim, squared = False) if l2_dist_proj_out else nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x):
        x = self.proj_in(x)
        x = F.gelu(x)
        return self.proj_out(x)

# main class

class L2DistanceTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        has_norms = True,
        attn_l2_dist_proj_out = False,
        ff_l2_dist_proj_out = False
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                l2_dist_proj_out = attn_l2_dist_proj_out
            )

            ff = FeedForward(
                dim,
                mult = ff_mult,
                l2_dist_proj_out = ff_l2_dist_proj_out
            )

            self.layers.append(ModuleList([
                RMSNorm(dim) if has_norms else nn.Identity(),
                attn,
                RMSNorm(dim) if has_norms else nn.Identity(),
                ff
            ]))

        self.norm = RMSNorm(dim) if has_norms else nn.Identity()

    def forward(
        self,
        x
    ):
        seq_len, device = x.shape[-1], x.device
        assert seq_len <= self.max_seq_len

        seq = torch.arange(seq_len, device = device)
        pos_emb = self.pos_emb(seq)

        # embed

        x = self.token_emb(x)
        x = x + pos_emb

        # main network

        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x)) + x
            x = ff(ff_norm(x)) + x

        x = self.norm(x)

        # use l2 distance back to original token embedding space for logits

        token_emb = self.token_emb.weight

        logits = -l2_dist_squared(x, token_emb)

        return logits
