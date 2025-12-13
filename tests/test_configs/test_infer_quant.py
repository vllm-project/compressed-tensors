# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from compressed_tensors.quantization import preset_name_to_scheme


@pytest.mark.parametrize(
    "preset,expected_format",
    [
        ["W8A8", "int-quantized"],
        ["W8A16", "pack-quantized"],
        ["W4A16", "pack-quantized"],
        ["FP8", "float-quantized"],
    ],
)
def test_infer_quant_format(preset, expected_format):
    quant_scheme = preset_name_to_scheme(preset, targets=["Linear"])
    assert quant_scheme.format == expected_format
