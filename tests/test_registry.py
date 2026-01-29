import pytest
from compressed_tensors import (
    BaseCompressor,
    BitmaskCompressor,
    BitmaskConfig,
    CompressionFormat,
    DenseCompressor,
    DenseSparsityConfig,
    SparsityCompressionConfig,
)


@pytest.mark.parametrize(
    "name,type",
    [
        [CompressionFormat.sparse_bitmask.value, BitmaskConfig],
        [CompressionFormat.dense.value, DenseSparsityConfig],
    ],
)
def test_configs(name, type):
    config = SparsityCompressionConfig.load_from_registry(name)
    assert isinstance(config, type)
    assert config.format == name


@pytest.mark.parametrize(
    "name,type",
    [
        [CompressionFormat.sparse_bitmask.value, BitmaskCompressor],
        [CompressionFormat.dense.value, DenseCompressor],
    ],
)
def test_compressors(name, type):
    compressor = BaseCompressor.load_from_registry(
        name, config=SparsityCompressionConfig(format="none")
    )
    assert isinstance(compressor, type)
    assert isinstance(compressor.config, SparsityCompressionConfig)
    assert compressor.config.format == "none"
