# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


__all__ = ["InternalModule"]


class InternalModule(torch.nn.Module):
    """
    Abstract base class for modules which are not a part of the the model definition.
    `torch.nn.Module`s which inherit from this class will not be targeted by configs

    This is typically used to skip apply configs to `Observers` and `Transforms`
    """

    pass
