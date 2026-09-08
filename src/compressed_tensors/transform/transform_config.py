# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.transform import TransformScheme
from compressed_tensors.utils import find_unique_name
from pydantic import BaseModel, ConfigDict


__all__ = ["TransformConfig"]


class TransformConfig(BaseModel):
    """
    Configuration of transforms to be applied to a model. This config is to be
    serialized within a model's `config.json` file

    :param config_groups: A dictionary of `TransformSchemes` that should be applied
        to a particular model. The keys can be any arbitrary string
    """

    config_groups: dict[str, TransformScheme]

    model_config = ConfigDict(extra="forbid")

    def merge(self, other: "TransformConfig") -> None:
        """
        Merge another TransformConfig into self in-place. Config groups from
        ``other`` are appended with unique keys to avoid collisions.
        """
        for key, transform in other.config_groups.items():
            unique_key = find_unique_name(key, self.config_groups.keys())
            self.config_groups[unique_key] = transform
