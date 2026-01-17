# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""EDM - Elucidating the Design Space of Diffusion-Based Generative Models"""

from .generate import edm_sampler, ablation_sampler, StackedRandomGenerator
from .training.networks import (
    EDMPrecond, VPPrecond, VEPrecond, iDDPMPrecond,
    SongUNet, DhariwalUNet,
    Linear, Conv2d, GroupNorm, UNetBlock,
    PositionalEmbedding, FourierEmbedding,
)
from .dnnlib import EasyDict
from .dnnlib.util import open_url
