from typing import Iterable, Iterator, Mapping

import torch
from compressed_tensors.utils.match import _match_name

__all__ = [
    "match_quantizable_tensors",
]


def match_quantizable_tensors(
    tensors: Mapping[str, torch.Tensor],
    ignore: Iterable[str],
    targets: Iterable[str] = tuple(),
    allow_nonquantizable: bool = False,
) -> Iterator[tuple[str, str]]:
    """
    Match all quantizable tensors that are not ignored and are
    targeted

    :param tensors: Mapping of name in safetensors file to actual tensor
    :param ignore: ignore individual tensor if any match is found with
        elements in ignore (regex allowed)
    :param targets: include if any match is found with elements in targets
        (regex allowed). Unlike model-based matching utils, this only checks
        names, not classes. If empty or if "Linear" is included, assume targets
        is all-inclusive.
    :param allow_nonquantizable: Override to include non-quantizable tensors,
        useful when performing other processing on tensors beyond quantization.

    :return: iterator of module_name and full tensor name meeting filtering
        criterion.
    """
    for name in list(tensors.keys()):
        module_name, param_name = name.rsplit(".", 1)

        is_quantizable = allow_nonquantizable or (
            (param_name == "weight") and not module_name.endswith("norm")
        )

        if len(targets) == 0 or "Linear" in targets:
            is_targeted = True
        else:
            is_targeted = any((_match_name(module_name, target)) for target in targets)

        is_ignored = any(_match_name(module_name, ign) for ign in ignore)

        if is_quantizable and is_targeted and not is_ignored:
            yield module_name, name
