# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


nbformat = pytest.importorskip("nbformat")
from nbconvert.preprocessors import ExecutePreprocessor  # noqa: E402


@pytest.mark.skip(
    reason="GHA not setup yet to run those tests. The test should work locally"
)
@pytest.mark.parametrize("notebook", ["examples/bitmask_compression.ipynb"])
def test_notebook_exec(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"
