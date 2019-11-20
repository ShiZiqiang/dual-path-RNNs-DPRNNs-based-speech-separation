#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 19-11-5 15:11:34
# @Author   : zm
# @File     : amazing.py
# @Software : PyCharm

import torch
from models import *


# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, input):
        input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input) # (B, E, L)-->(B, N, L)
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B, N, L, K: L is the segment_size
        #print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPRNN
        output = self.DPRNN(enc_segments).view(batch_size * self.num_spk, self.feature_dim, self.segment_size,
                                                   -1)  # B*nspk, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*nspk, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nspk, K, T
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_spk, -1,
                                                                self.feature_dim)  # B, nspk, T, N

        return bf_filter


# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250,
                 nspk=2, win_len=2):
        super(FaSNet_base, self).__init__()

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        # waveform encoder
        #self.encoder = nn.Conv1d(1, self.enc_dim, self.feature_dim, bias=False)
        self.encoder = Encoder(win_len, enc_dim) # [B T]-->[B N L]
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8) # [B N L]-->[B N L]
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim,
                                self.num_spk, self.layer, self.segment_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        input = input.to(device)
        B, _ = input.size()
        # mixture, rest = self.pad_input(input, self.window)
        #print('mixture.shape {}'.format(mixture.shape))
        mixture_w = self.encoder(input)  # B, E, L

        score_ = self.enc_LN(mixture_w) # B, E, L
        #print('mixture_w.shape {}'.format(mixture_w.shape))
        score_ = self.separator(score_)  # B, nspk, T, N
        #print('score_.shape {}'.format(score_.shape))
        score_ = score_.view(B*self.num_spk, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nspk, N, T
        #print('score_.shape {}'.format(score_.shape))
        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        #print('score.shape {}'.format(score.shape))
        score = score.view(B, self.num_spk, self.enc_dim, -1)  # [B*nspk, E, L] -> [B, nspk, E, L]
        #print('score.shape {}'.format(score.shape))
        est_mask = F.relu(score)

        est_source = self.decoder(mixture_w, est_mask) # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]

        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]

        return est_source


def test_model(model):
    x = torch.rand(2, 32000)  # (batch, num_mic, length)
    y = model(x)
    print(y.shape)  # (batch, nspk, length)

if __name__ == "__main__":
    model_origin = FaSNet_base(enc_dim=256, feature_dim=64, hidden_dim=128, layer=6, segment_size=250,
                                 nspk=2, win_len=2)
    test_model(model_origin)

