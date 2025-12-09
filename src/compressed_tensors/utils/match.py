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

import logging
import os
import re
from collections import defaultdict
from collections.abc import Generator
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import torch
from compressed_tensors.utils.internal import InternalModule


_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "match_named_modules",
    "match_named_parameters",
    "match_targets",
    "match_modules_set",
    "get_lowest_common_ancestor_name",
    "is_match",
    "is_narrow_match",
]


FusedMappping = Mapping[str, Iterable[str]]


def match_named_modules(
    model: torch.nn.Module,
    targets: Optional[Iterable[str]],
    ignore: Optional[Iterable[str]] = None,
    fused: Optional[FusedMappping] = None,
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module]]:
    """
    Yields names and modules which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing submodules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards. See `compressed_tensors.utils.match.is_match`
    :param warn_on_fail: if True, warns if any targets do not match any modules in model
    :return: generator of module names and modules
    """
    targets = targets or []
    ignore = ignore or []

    unmatched_targets = set(targets)

    for name, module in model.named_modules():
        for target in targets:
            if is_match(name, module, target, fused=fused):
                unmatched_targets -= {target}
                if not is_match(name, module, ignore, fused=fused):
                    yield name, module
                break

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_named_parameters(
    model: torch.nn.Module,
    targets: Optional[Iterable[str]],
    ignore: Optional[Iterable[str]] = None,
    fused: Optional[FusedMappping] = None,
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module, torch.nn.Parameter]]:
    """
    Yields parameters which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing params to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards. See `compressed_tensors.utils.match.is_match`
    :param warn_on_fail: if True, warns if any targets do not match any params in model
    :return: generator of fully-qualified param names, parent modules, and params
    """
    targets = targets or []
    ignore = ignore or []

    unmatched_targets = set(targets)
    for module_name, module in model.named_modules():
        if isinstance(module, InternalModule):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            param_fqn = f"{module_name}.{param_name}"
            for target in targets:
                if _match_name(param_fqn, target, fused):
                    unmatched_targets -= {target}

                    if not any(_match_name(param_fqn, ign, fused) for ign in ignore):
                        yield param_fqn, module, param

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_targets(
    name: str, module: torch.nn.Module, targets: Optional[Iterable[str]]
) -> List[str]:
    """
    Returns the targets that match the given name and module.

    :param name: the name of the module
    :param module: the module to match
    :param targets: the target strings, potentially containing "re:" prefixes
    :return: the targets that match the given name and module

    Outputs are ordered by type: exact name match, regex name match, class name match
    """
    targets = targets or []

    if isinstance(module, InternalModule):
        return []

    # The order of the output `matches` list matters, they are arranged from most
    # specific to least specific, and this order will be used when merging configs.
    # The entries are sorted in the following order:
    #     1. matches on exact strings
    #     2. matches on regex patterns
    #     3. matches on module names (e.g. "Linear")

    targets = sorted(targets, key=lambda x: ("re:" in x, x))
    matched_targets = []
    for target in targets:
        if _match_name(name, target):
            matched_targets.append(target)

    for target in targets:
        if _match_class(module, target) and target not in matched_targets:
            matched_targets.append(target)

    return matched_targets


def get_lowest_common_ancestor_name(names: list[str | None]) -> str:
    """
    Given a list of names, returns the lowest-scope common name ignoring Nones.

    Implementation is a small alteration of os.path.commonprefix
    https://docs.python.org/3/library/os.path.html#os.path.commonprefix

    ([s1, s2]->prefix->result)
    # case 0: multiple modules: [abc.a., abc.b.] -> .abc. -> abc
    # case 1: single module: [abc.] -> .abc. -> abc
    # case 2: substring modules: [abc., ab.] -> .ab -> ""
    # case 3: parent & child: [ab., ab.a.] -> .ab. -> ab
    """
    names = [name for name in names if name is not None]
    if len(names) == 0:
        return ""

    # 1) find longest shared prefix
    s1 = "." + min(names) + "."
    s2 = "." + max(names) + "."
    common_prefix = os.path.commonprefix([s1, s2])
    # 2) throw away right most dot and name fragment, throw away leftmost char
    # ".keep.thro" -> "keep", "." -> ""
    return common_prefix[1 : common_prefix.rfind(".")]


def match_modules_set(
    model: torch.nn.Module,
    targets: Optional[Iterable[str]],
    ignore: Optional[Iterable[str]] = None,
    error_on_module_rematch: bool = True,
) -> Generator[List[List[torch.nn.Module]]]:
    """
    Yields modules grouped by parent context.

    We group by parent context so that we can return ALL matches of a
    specific target that can be paired with another target. This is most
    relevant in the case of MoE modules with multiple modules for each
    expert i.e. post_attention_layernorm <-> mlp.expert.N.gate_proj,
    mlp.expert.N.up_proj for all N. The parent context will differ from
    one layer to another while being the same for one expert to another.

    Each returned group is a list (of lists) with the same size
    and order as `targets` while all matches for each target and
    the overall order of the groups are ordered in the same way
    as `model.named_modules`


    E.g. the following targets would yield modules belonging to the following layers:
    ```python3
    match_modules_set(model, ["q_proj", "k_proj", "v_proj"]) == (
        [
            [`layers.0.self_attn.q_proj`],
            [`layers.0.self_attn.k_proj`],
            [`layers.0.self_attn.v_proj`],
        ],
        [
            [`layers.1.self_attn.q_proj`],
            [`layers.1.self_attn.k_proj`],
            [`layers.1.self_attn.v_proj`],
        ],
        ...
    )
    ```

    This can be used to match layers to their corresponding downstream counterparts.
    For example, matching layer norms to their subsequent linear layers
    ```python3
    for norm, q, k, v in match_modules_set(model, (norm_tgt, q_tgt, k_tgt, v_tgt)):
        fuse_norm_linears(*norm, [*q, *k, *v])
    ```

    Alternatively for MoE you would get multiple matches
    per target per group, E.g.

    ```python3

    targets = [
        "post_attention_layernorm",
        "up_proj",
        "down_proj"
    ]
    match_modules_set(model, targets) == (
        [
            [layers.0.post_attention_layernorm],
            [
                `layers.0.mlp.experts.0.up_proj`,
                `layers.0.mlp.experts.1.up_proj`,
                ...
            ],
            [
                `layers.0.mlp.experts.0.down_proj`,
                `layers.0.mlp.experts.1.down_proj`,
                ...

            ]
        ], # <- first yield
        [
            [layers.1.post_attention_layernorm],
            [
                `layers.1.mlp.experts.0.up_proj`,
                `layers.1.mlp.experts.1.up_proj`,
                ...
            ],
            [
                `layers.1.mlp.experts.0.down_proj`,
                `layers.1.mlp.experts.1.down_proj`,
                ...
            ]
        ],
        ...
    )
    ```

    :param model: model containing modules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :param error_on_module_rematch: if True, errors when a module gets
      matched to multiple targets, if False, no error. (Defaults to True)
    """
    targets = targets or []
    ignore = ignore or []

    # as we iterate through modules and try to match them with targets,
    # the algorithm can be in 2 possible states:
    # 0) unmatched_targets > 0, i.e. some of the targets haven't been matched.
    #   Keep matching until all targets have at least one match
    # 1) unmatched_targets == 0 i.e. we have at least one match for each target.
    #   At this point we are unsure if we have a full set or if we need to add
    #   more matches.
    # There are 3 things that can happen once were in state 1:
    # A) found a new match with same parent_context,
    #   (add it to matches and keep going)
    # B) found a new match with different parent_context, i.e. we found a match
    #   that requires a deeper parent context, this indicates that this match
    #   should be part of a new set.
    #   (yield current set [not including newest match] and go back to state 0)
    # C) ran out of modules, we will always yield the final remaining set when
    #   we we've iterated through all the modules in the model.
    #   (yield final set then exit.)
    # Note: its possible to iterate through all the modules in the model while
    #   not having a full matched set if the user specified a bad matching, in
    #   that case something has gone wrong and we error
    matches = defaultdict(list)
    parent_context = None
    unmatched_targets = set(targets)

    for name, module in model.named_modules():
        matched_targets_for_cur_module = set()
        for target in targets:
            if is_match(name, module, target, ignore):
                new_parent_context = get_lowest_common_ancestor_name(
                    [name, parent_context]
                )

                # code for (B)
                if not unmatched_targets and new_parent_context != parent_context:
                    yield [matches[target] for target in targets]
                    matches = defaultdict(list)
                    new_parent_context = name
                    unmatched_targets = set(targets)

                matches[target].append(module)
                parent_context = new_parent_context
                unmatched_targets -= {target}
                matched_targets_for_cur_module |= {target}

        if len(matched_targets_for_cur_module) > 1 and error_on_module_rematch:
            raise ValueError(
                f"module: {name} was matched with multiple targets: "
                f"{matched_targets_for_cur_module} which is unexpected "
                "disable this check by setting `error_on_module_rematch = False`"
            )

    # never found anything
    if unmatched_targets == set(targets):
        return

    # code for (C)
    if not unmatched_targets:  # have a full matching
        yield [matches[target] for target in targets]
        return

    raise ValueError(
        f"Found a final incomplete set with matches found for keys: "
        f"{set(targets) - unmatched_targets} "
        f"but no matches found for keys: {unmatched_targets}"
    )


def is_match(
    name: str,
    module: torch.nn.Module,
    targets: Union[str, Iterable[str]],
    ignore: Union[str, Iterable[str]] = tuple(),
    fused: Optional[FusedMappping] = None,
) -> bool:
    """
    Returns true if either module name or module parent classes match against target
    and the module is not an internal module. The name and module may refer to a fused
    module defined by vLLM. In these cases, a `fused` mapping must be provided.

    For example, in `vllm/model_executor/models/llama.py`:
    ```python
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    ```

    :param name: name of module
    :param module: module to match
    :param target: target which matches name or module, potentially contains regex
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards
    """
    targets = [targets] if isinstance(targets, str) else targets
    ignore = [ignore] if isinstance(ignore, str) else ignore

    return not isinstance(module, InternalModule) and (
        any(
            _match_name(name, target, fused) or _match_class(module, target)
            for target in targets
        )
        and not any(
            _match_name(name, ign, fused) or _match_class(module, ign) for ign in ignore
        )
    )


def is_narrow_match(
    model: torch.nn.Module,
    targets: Union[str, Iterable[str]],
    name: str,
    module: Optional[torch.nn.Module] = None,
) -> bool:
    """
    Checks if any of the targets narrowly match the module. A target narrowly matches
    a module if the target matches the module, but does not match the module's parent

    :param model: model containing both module and its parent
    :param targets: target strings, potentially containing "re:" prefixes
    :param name: name of module to match
    :param module: module to match. If none is provided, then get module from model
    :return: True if any of the targets narrow match the module
    """
    targets = [targets] if isinstance(targets, str) else targets
    module = module if module is not None else model.get_submodule(name)

    parent_name = name.rsplit(".", 1)[0]
    parent = model.get_submodule(parent_name)

    return any(
        is_match(name, module, target) and not is_match(parent_name, parent, target)
        for target in targets
    )


def _match_name(name: str, target: str, fused: Optional[FusedMappping] = None) -> bool:
    """
    Returns true if target string begins with "re:" and regex matches or if target
    string exactly matches name. If the name refers to a fused module defined by vLLM,
    a `fused` mapping must be provided.

    :param name: name of module
    :param target: target name, potentially contains regex
    :fused: optional mapping from suffixes of fused modules to the suffixes of their
        corresponding shards
    """
    if fused is not None:
        for fused_suffix in fused:
            if name.endswith(fused_suffix):
                name_stripped = name.removesuffix(fused_suffix)
                return any(
                    _match_name(name_stripped + shard_suffix, target)
                    for shard_suffix in fused[fused_suffix]
                )

    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    else:
        return target == name


def _match_class(module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if any torch parent class names match the target string exactly.
    A special exception is made for vllm's `LinearBase` class which matches `Linear`

    :param module: module to match
    :param target: target which matches name or module
    """
    # will never match against a regex pattern since `:` is not allowed in class names
    return any(
        (
            issubclass(cls, torch.nn.Module)
            and (
                cls.__name__ == target
                or (cls.__name__ == "LinearBase" and target == "Linear")
            )
        )
        for cls in module.__class__.__mro__
    )
