from compressed_tensors.transform import TransformScheme
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
