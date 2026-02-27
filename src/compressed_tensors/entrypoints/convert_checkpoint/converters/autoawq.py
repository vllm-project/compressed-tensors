# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert_checkpoint.converters import Converter


# TODO implement
class AutoAWQConverter(Converter):
    """
    Convert params from AutoAWQ W4A16 to CT W4A16 convention
    """

    pass
