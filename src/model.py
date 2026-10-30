"""DeepLOB (Zhang, Zohren & Roberts, 2019).

Layer for layer the PyTorch version from the authors' repository, with one
fix: it returns logits. The official code applies softmax and then
CrossEntropyLoss, which applies softmax again.

Shapes for a (B, 1, 100, 40) input:
    conv blocks  -> (B, 32, 82, 1)   time shrinks 100 -> 82, the 40 LOB columns collapse to 1
    inception    -> (B, 192, 82, 1)  three 64-channel branches concatenated
    LSTM         -> (B, 64)          last hidden state
    linear       -> (B, 3)           logits for up / stationary / down
"""

from __future__ import annotations

import torch
from torch import nn


def _conv(c_in: int, c_out: int, kernel: tuple[int, int], activation: type[nn.Module], **kwargs) -> list[nn.Module]:
    act = activation(negative_slope=0.01) if activation is nn.LeakyReLU else activation()
    return [nn.Conv2d(c_in, c_out, kernel, **kwargs), act, nn.BatchNorm2d(c_out)]


def _conv_block(c_in: int, first_kernel: tuple[int, int], first_stride: tuple[int, int],
                activation: type[nn.Module]) -> nn.Sequential:
    """One wide conv that merges LOB columns, then two (4, 1) convs along time."""
    return nn.Sequential(
        *_conv(c_in, 32, first_kernel, activation, stride=first_stride),
        *_conv(32, 32, (4, 1), activation),
        *_conv(32, 32, (4, 1), activation),
    )


class DeepLOB(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv1 = _conv_block(1, (1, 2), (1, 2), nn.LeakyReLU)   # pairs price with volume
        self.conv2 = _conv_block(32, (1, 2), (1, 2), nn.Tanh)       # pairs ask with bid
        self.conv3 = _conv_block(32, (1, 10), (1, 1), nn.LeakyReLU) # merges the 10 levels

        self.inp1 = nn.Sequential(*_conv(32, 64, (1, 1), nn.LeakyReLU, padding="same"),
                                  *_conv(64, 64, (3, 1), nn.LeakyReLU, padding="same"))
        self.inp2 = nn.Sequential(*_conv(32, 64, (1, 1), nn.LeakyReLU, padding="same"),
                                  *_conv(64, 64, (5, 1), nn.LeakyReLU, padding="same"))
        self.inp3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
                                  *_conv(32, 64, (1, 1), nn.LeakyReLU, padding="same"))

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)  # (B, 192, T, 1)
        x = x.squeeze(3).transpose(1, 2)                                    # (B, T, 192)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1])
