# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from compressed_tensors.registry import RegistryMixin


@pytest.mark.parametrize(
    "filename", ["custom_plugin.py", "CustomPlugin.py", "custom plugin.py"]
)
def test_load_plugin_preserves_path_and_class_name(tmp_path, filename):
    plugin = tmp_path / filename
    plugin.write_text(
        "class CustomPlugin:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n"
    )
    instance = RegistryMixin.load_from_registry(f"{plugin}:CustomPlugin", value=17)
    assert instance.value == 17
    assert type(instance).__name__ == "CustomPlugin"


def test_registered_names_still_normalize():
    class Parent(RegistryMixin):
        pass

    @Parent.register(name="My_Plugin")
    class Plugin:
        pass

    assert isinstance(Parent.load_from_registry("MY PLUGIN"), Plugin)
