# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

from compressed_tensors.config import MoESparsityMethod
from pydantic import BaseModel, StrictInt


__all__ = ["ExpertSparsityConfig"]


class ExpertSparsityConfig(BaseModel):
    """
    Configuration for storing a Mixture-of-Experts (MoE) model with expert
    sparsity, where the number of experts is reduced on a per-layer basis.

    Two methods of inducing expert sparsity are supported via ``method``:

    * ``"prune"`` (e.g. REAP): low-value experts are pruned entirely. Retained
      experts are left-shifted in the checkpoint.
    * ``"novice"`` (e.g. MoNe): experts are collapsed into constant "novice"
      experts. Retained experts are left-shifted and novices remain as a single
      tensor.

    The ``num_experts`` field encodes the per-layer expert count. It is a list
    whose length equals the total number of decoder layers in the model. Each
    entry is either:

    * ``None`` for a dense (non-MoE) layer, or
    * an ``int`` giving the number of experts retained in that MoE layer.

    Example (REAP / ``prune`` or MoNe / ``novice``)::

        {
            "method": "prune",
            "num_experts": [
                null,  # dense layers
                ...
                129,   # per-layer sparsity
                253,
                128,
                128,
                ...
            ]
        }

    :param method: method used to induce expert sparsity, one of
        :class:`MoESparsityMethod` ("prune" or "novice")
    :param num_experts: per-layer retained expert counts. ``None`` entries
        denote dense layers; integer entries denote the number of experts in
        that MoE layer
    """

    method: MoESparsityMethod
    num_experts: List[Optional[StrictInt]]
