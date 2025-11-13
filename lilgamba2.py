from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from scans import selective_scan
from sample_utils import geometric_dist


@dataclass
class GambaArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'

    num_gamba: int = 8
    decay_rate: float = 0.5

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class LilGamba(nn.Module):
    def __init__(self, args: GambaArgs):
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([GambaResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        h_queue = []

        for layer_count, layer in enumerate(self.layers, start=1):
            h_input = [self.norm_f(h) for h in h_queue] if h_queue else []
            h_input = [x] + h_input

            x, new_hidden_list = layer(h_input)

            # Prepend newest hidden states directly to the queue
            for h_new in new_hidden_list:
                if len(h_queue) < layer.args.num_gamba:
                    h_queue = [h_new] + h_queue
                else:
                    prob_dist = geometric_dist(
                        v=1.0 - layer.args.decay_rate * layer_count / (layer_count + 1),
                        N=layer.args.num_gamba
                    )
                    idx_to_replace = torch.multinomial(
                        torch.tensor(prob_dist, device=x.device), 1
                    ).item()
                    h_queue.pop(idx_to_replace)
                    h_queue = [h_new] + h_queue

        x = self.norm_f(x)
        return self.lm_head(x)


class GambaResidualBlock(nn.Module):
    def __init__(self, args: GambaArgs):
        super().__init__()
        self.args = args
        self.mixer = GambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, h_list):
        h_input = [self.norm(h) for h in h_list]
        x, new_hidden_list = self.mixer(h_input)
        x = x + h_list[0]  # Residual connection
        return x, new_hidden_list


class GambaBlock(nn.Module):
    def __init__(self, args: GambaArgs):
        super().__init__()
        self.args = args
        self.params_list = [GambaParams(args) for _ in range(args.num_gamba)]

    def forward(self, h_list):
        assert len(h_list) > 0, "h_list must contain at least one tensor"
        (b, l, d) = h_list[0].shape
        output = torch.zeros((b, l, d), device=h_list[0].device, dtype=h_list[0].dtype)
        new_hidden_list = []

        for i, params in enumerate(self.params_list):
            if i >= len(h_list):
                break
            h_i = h_list[i]

            x_proj = params.in_proj(h_i)
            x, res = x_proj.split([self.args.d_inner, self.args.d_inner], dim=-1)
            x = rearrange(x, 'b l d -> b d l')
            x = params.conv1d(x)[:, :, :l]
            x = rearrange(x, 'b d l -> b l d')
            x = F.silu(x)

            y = self.ssm(x, params)
            y_hidden = y * F.silu(res)
            y_proj = params.out_proj(y_hidden)

            new_hidden_list.append(y_proj)  # newest first
            output += y_proj

        return output, new_hidden_list

    def ssm(self, x, params):
        d_in, n = params.A_log.shape
        A = -torch.exp(params.A_log.float())
        D = params.D.float()
        x_dbl = params.x_proj(x)
        delta, B, C = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(params.dt_proj(delta))
        return selective_scan(x, delta, A, B, C, D, mode=self.args.scan_mode)


class GambaParams:
    def __init__(self, args: GambaArgs):
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
