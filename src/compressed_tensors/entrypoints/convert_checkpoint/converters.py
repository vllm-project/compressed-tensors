from typing import Iterable, Protocol

import torch

from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert_checkpoint.helpers import match_quantizable_tensors
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, NVFP4

class Converter(Protocol):
    """
    Converter interface, to modify safetensors files based 
    on tensor name and pointer to torch.Tensor
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        pass

    def validate(self, tensors: dict[str, torch.Tensor]):
        pass

    def create_scheme(self) -> QuantizationScheme:
        pass 



class ModelOptNvfp4Converter(Converter):
    """
    Convert params from modelopt NVFP4 to CT NVFP4 convention
    """

    def __init__(
        self,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple(),
    ):
        self.ignore = ignore
        self.targets = targets

    def process(self, tensors: dict[str, torch.Tensor]):
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]

            match param_name:
                # input_scale -> input_global_scale F32
                case "input_scale":
                    # convert modelopt input_scale x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1070-L1073
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1134
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L190
                    tensors[f"{module_name}.input_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                # weight -> weight_packed U8
                case "weight":
                    tensors[f"{module_name}.weight_packed"] = tensors[name]
                    del tensors[name]
                # weight_scale -> weight_scale F8_E4M3
                case "weight_scale":
                    pass
                # weight_scale_2 -> weight_global_scale F32
                case "weight_scale_2":
                    # convert modelopt weight_scale_2 x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1066-L1068
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L163-L166
                    tensors[f"{module_name}.weight_global_scale"] = 1 / tensors[name]
                    del tensors[name]


    def validate(self, tensors: dict[str, torch.Tensor]):
        for _, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]

            if param_name not in (
                "input_scale",
                "weight",
                "weight_scale",
                "weight_scale_2",
                # NOTE: some models like nvidia/Qwen3-32B-NVFP4 have other params
                # that just need to be passed through. 
                # TODO Maybe convert this to a warning_once instead of error
                "k_scale",
                "v_scale"
            ):
                raise RuntimeError(f"Hit unexpected tensor {name}")

    def create_scheme(self) -> QuantizationScheme:
        return QuantizationScheme(
            **NVFP4,
            targets=self.targets,
            format=CompressionFormat.nvfp4_pack_quantized.value,
        )


#TODO implement
class AutoAWQConverter(Converter):
    """
    Convert params from AutoAWQ W4A16 to CT W4A16 convention
    """
    pass
