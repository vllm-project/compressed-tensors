# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme


@pytest.fixture
def create_quantization_scheme():
    def quantization_scheme(
        targets: List[str],
        weights: Optional[QuantizationArgs] = None,
        input_activations: Optional[QuantizationArgs] = None,
        output_activations: Optional[QuantizationArgs] = None,
    ):
        return QuantizationScheme(
            targets=targets,
            weights=weights,
            input_activations=input_activations,
            output_activations=output_activations,
        )

    return quantization_scheme


@pytest.fixture
def mock_frozen():
    def update_status(model: torch.nn.Module):
        model.quantization_status = QuantizationStatus.FROZEN

    return update_status
