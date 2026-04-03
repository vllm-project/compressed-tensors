# `compressed_tensors` Offload Module

## Overview

The `compressed_tensors.offload` module provides a transparent, flexible system for offloading model weights between devices (CPU, GPU, and disk) to enable inference with models larger than available GPU memory. It was designed with the following goals:

- **Transparency**: offloading logic is invisible to the model's forward pass — no code changes needed in model implementations.
- **Flexibility**: supports CPU offload, disk offload, multi-GPU dispatch, and distributed (multi-process) setups via a single unified interface.
- **Performance**: lazy onloading of individual parameters and buffers means that device movement workload is minimized, while the provided utilities encourage keeping offloaded tensors on their offloaded device. 
- **Interoperability**: can convert to and from HuggingFace `accelerate` offloading, making it compatible with `transformers` loading and saving workflows.

---

## Core Philosophy

### The OffloadCache: Replacing `_parameters` and `_buffers`

The central design decision is that offloading is implemented by **replacing a `torch.nn.Module`'s `_parameters` and `_buffers` dictionaries** with `OffloadCache` instances.

`OffloadCache` is a `MutableMapping` that behaves like a dict, but with two key differences:

- **On write (`__setitem__`):** the tensor is immediately offloaded to its storage device (CPU, disk, etc.).
- **On read (`__getitem__`):** the tensor is onloaded (brought back to the execution device) before being returned.

Because PyTorch accesses module parameters through `module._parameters[name]` during a forward pass, this substitution is fully transparent — PyTorch naturally triggers onloading whenever it accesses a weight, and offloading occurs automatically when weights are assigned or after the forward pass completes.

```
Normal module:                            Offloaded module:
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ module._parameters             │        │ module._parameters             │
│   = { "weight": <cuda tensor>} │   →    │   = OffloadCache {             │
└────────────────────────────────┘        │       offload: cpu             │
                                          │       onload: cuda:0           │
                                          │       offloaded_values: {      │
                                          │        "weight": <cpu tensor>  │
                                          │       }                        │
                                          │     }                          │
                                          └────────────────────────────────┘
```

### The Forward Wrapper

In addition to replacing `_parameters` and `_buffers`, `offload_module` wraps the module's `forward` method. The wrapper moves all input tensors to the `onload_device` before the forward call. This ensures that activations and weights are on the same device without requiring the model's own code to manage device placement.

### Onload/Offload Lifecycle

During a normal forward pass:
1. Input tensors arrive on any device.
2. The forward wrapper moves inputs to `onload_device`.
3. PyTorch accesses parameters via `_parameters[name]` → `OffloadCache.__getitem__` → `onload()` is called, moving weights to device.
4. The forward computation runs entirely on `onload_device`.
5. After the forward pass, onloaded weights are released (garbage collected) unless `disable_offloading` is active.

### Global Control Flags

Two class-level flags on `OffloadCache` allow callers to suppress device movement across all modules simultaneously:

- `offloading_disabled`: weights that have been onloaded are kept in memory; re-accesses are cache hits.
- `onloading_disabled`: reads and writes bypass device movement entirely (useful for inspecting raw offloaded tensors).

These are exposed as context managers: `disable_offloading()` and `disable_onloading()`.

---

## Module Reference

### `cache/` — OffloadCache Implementations

#### `OffloadCache` (base class) — `cache/base.py`

The abstract `MutableMapping` that all cache implementations extend.

```python
class OffloadCache(MutableMapping, ABC):
    onload_device: torch.device | str
    offload_device: torch.device | Literal["disk"]
```

**Key methods:**

| Method | Description |
|---|---|
| `cls_from_device(device)` | Returns the correct `OffloadCache` subclass for the given offload device and distributed state. Automatically selects distributed variants when `dist` is initialized. |
| `from_mapping(mapping, onload_device, **kwargs)` | Class method. Constructs an `OffloadCache` from an existing dict (e.g., `module._parameters`), offloading all values immediately. This is the canonical way to attach a cache to a module. |
| `onload(offloaded)` | *(abstract)* Given an offloaded tensor, returns the onloaded version. |
| `offload(tensor)` | *(abstract)* Given a tensor, offloads it to the storage device and returns a reference. |
| `update_offload(offloaded, data)` | *(abstract)* In-place update of an already-offloaded tensor without creating a new storage location. |
| `__getitem__(key)` | Onloads the tensor. Uses the onloaded cache if `offloading_disabled` is set. |
| `__setitem__(key, value)` | If the key exists and sizes match, calls `update_offload` (in-place update). Otherwise calls `offload` and stores the new reference. |
| `disable_offloading()` | *Context manager.* Onloaded tensors are cached in memory; subsequent reads are cache hits. All cached values are released on exit. |
| `disable_onloading()` | *Context manager.* All reads return offloaded tensors directly; writes assign directly without triggering device movement. Mainly for debugging. |

**When to use `disable_offloading`:**

```python
# Without: each access triggers a CPU→GPU copy
for _ in range(3):
    result = module.weight  # 3 separate copies

# With: first access copies, subsequent reads hit cache
with disable_offloading():
    for _ in range(3):
        result = module.weight  # 1 copy, 2 cache hits
```

**When to use `disable_onloading`:**

```python
# Inspect the raw offloaded tensor (e.g., CPU tensor) without triggering a copy
with disable_onloading():
    cpu_tensor = module.weight   # returns the CPU tensor directly
    module.weight = new_tensor   # assigns without triggering offload
```

---

#### `CPUCache` — `cache/cpu.py`

Offloads tensors to CPU RAM. Onloading is a standard `.to(device)` call from CPU to the configured `onload_device` (typically a CUDA device).

- **offload**: moves tensor to `torch.device("cpu")`.
- **onload**: moves tensor from CPU to `onload_device`.
- **update_offload**: `offloaded.copy_(data)` in-place on the CPU tensor.

**Use when:** GPU VRAM is insufficient and CPU RAM is available. This is the most common offload strategy.

---

#### `DeviceCache` — `cache/device.py`

Offloads tensors to a CUDA device. Onloading is typically a no-op (the tensor is already on device), but handles the case where `onload_device` is changed after initialization (e.g., during `set_onload_device` reconfiguration).

- **offload**: moves tensor to the device (`self.offload_device = self.onload_device` at init).
- **onload**: `send_tensors(offloaded, device=self.onload_device)`.
- **update_offload**: in-place copy.

**Use when:** a module is permanently resident on a device and you want consistent management via the `OffloadCache` interface (e.g., when `dispatch_model` keeps some modules fully on-device).

---

#### `DiskCache` — `cache/disk.py`

Offloads tensors to disk as `safetensors` files. In-memory, offloaded tensors are represented as **meta tensors** (tensors with no data, only shape/dtype/stride). A separate `index` dict maps meta tensor identity → disk location.

- **offload**: writes tensor to a `.safetensors` file in `offload_dir`, returns a meta tensor.
- **onload**: reads the file at the stored path with `safe_open`, reconstructs the tensor.
- **update_offload**: overwrites the file with new data.
- **`__delitem__`**: removes from index and deletes the file if it was created (not the original model file).

**Use when:** neither CPU RAM nor GPU VRAM can hold all weights. Disk offload is the slowest but allows running arbitrarily large models.

> **Note:** `DiskCache` requires weights to be stored in `safetensors` format. When loading from non-safetensors formats, weights are onloaded and re-saved.

---

#### Distributed Cache Variants

Each of the three non-distributed cache types has a distributed counterpart that synchronizes data across ranks during `offload`.

##### `DistributedCPUCache` — `cache/dist_cpu.py`

Extends `CPUCache`. On `offload`:

1. Rank 0 creates a CPU tensor and calls `.share_memory_()` to place it in POSIX shared memory (`/dev/shm`).
2. Rank 0 broadcasts the shared memory file handle to all other ranks.
3. All other ranks reconstruct the tensor by attaching to the same shared memory.
4. A `dist.barrier()` ensures rank 0 does not GC the memory before others attach.

**Result:** all ranks share the same physical CPU memory for offloaded weights — no redundant copies.

**Use when:** running multi-process inference and offloading to CPU. The shared memory avoids `N × model_size` RAM usage.

---

##### `DistributedDeviceCache` — `cache/dist_device.py`

Extends `DeviceCache`. On `offload`:

1. Rank 0 moves the tensor to its device.
2. All other ranks create an empty tensor on their local device.
3. `dist.broadcast(as_broadcastable(tensor), src=0)` sends the data to all ranks.

**Result:** each rank has its own copy of the tensor on its local GPU. The model is **replicated** across devices.

**Use when:** running data-parallel multi-GPU inference where all ranks need identical weights.

---

##### `DistributedDiskCache` — `cache/dist_disk.py`

Extends `DiskCache`. On `offload`:

1. Rank 0 writes the safetensors file and broadcasts the file path + weight name + dtype.
2. Other ranks construct a meta tensor and populate their `index` with the broadcasted path.
3. `dist.barrier()` waits for the write to complete before all ranks proceed.

**Result:** all ranks share the same disk files for offloaded weights.

---

### `module.py` — Module-Level Offloading

#### `offload_module(module, onload_device, offload_device, **kwargs)`

The primary function for attaching offloading to a single `torch.nn.Module`. It:

1. Selects the correct `OffloadCache` subclass via `OffloadCache.cls_from_device(offload_device)`.
2. Replaces `module._parameters` and `module._buffers` with cache instances (offloading all existing tensors).
3. Wraps `module.forward` to move input tensors to `onload_device` before the forward call.
4. Stores the original forward function as `module._original_forward_func` so it can be restored.

```python
# Offload a single linear layer to CPU, execute on cuda:0
offload_module(layer, onload_device="cuda:0", offload_device="cpu")
```

**When to use:** when you want fine-grained control over which specific modules are offloaded. For model-wide dispatch, prefer `dispatch_model` or `set_onload_device`.

> **Note:** Raises `ValueError` if the module is already offloaded. Call `remove_module_offload` first.

---

#### `remove_module_offload(module, onload_tensors=False)`

Removes offloading from a single module, restoring plain dicts for `_parameters` and `_buffers` and restoring the original forward function.

- `onload_tensors=True`: restores weights to the `onload_device` (GPU). Use before running a forward pass without offloading.
- `onload_tensors=False` (default): keeps weights on the offload device. Use when freeing memory or re-dispatching.

---

#### `unwrap_offload_forward(module)` *(context manager)*

Temporarily removes the offload forward wrapper so the underlying forward function can be modified (e.g., by another library). Upon exiting, the offload wrapper is re-applied around any modifications made to `module.forward`.

```python
with unwrap_offload_forward(module):
    module.forward = my_patched_forward  # patching the real forward
# on exit: offload wrapper is applied around my_patched_forward
```

**Use when:** other code (e.g., PEFT, quantization hooks) needs to wrap the module's real forward without being wrapped around or interfering with the offload wrapper.

---

### `dispatch.py` — Model-Level Dispatch

#### `dispatch_model(model, device_memory=None, extra_memory=None, no_split_modules=None)`

The highest-level dispatch function. Automatically determines how to distribute a model across all available CUDA devices, maximizing GPU utilization with a CPU offload fallback.

**Algorithm:**
1. Queries all available CUDA devices for their total memory.
2. Uses `get_module_sizes` to measure each non-splittable module.
3. Performs a binary search to find the maximum `extra_memory` (reserved for activations/KV cache) such that all modules fit on device.
4. If the model fits entirely on GPU, dispatches with no offloading.
5. If not, falls back to placing as many modules as possible on GPU and offloading the rest to CPU.

```python
# Dispatch to all GPUs, reserve memory for KV cache automatically
model = dispatch_model(model)

# Dispatch with explicit device memory constraints
model = dispatch_model(model, device_memory={torch.device("cuda:0"): 16e9})
```

**Parameters:**
- `device_memory`: optional mapping of device → available bytes. Defaults to querying all CUDA devices.
- `extra_memory`: bytes to reserve for activations. If `None`, estimated from model config.
- `no_split_modules`: names of module classes that cannot be split across devices. Defaults to `model._no_split_modules` if available.

**When to use:** the primary entry point for production deployment — automatic, memory-aware, multi-GPU dispatch.

---

#### `set_onload_device(model, onload_device)`

A lighter-weight dispatch that moves all modules in a model to the same `onload_device`, without changing where weights are stored. For modules not yet offloaded, it offloads them to their current device.

```python
# Move all execution to cuda:0, keeping offloads unchanged
model = set_onload_device(model, onload_device="cuda:0")
```

**When to use:** when you have already loaded a model with weights in the right place (e.g., via `load_offloaded_model`) and just need to set the execution device. Less powerful than `dispatch_model` but simpler.

> **Note:** `offload_model` is a deprecated alias for this function.

---

#### `dispatch_with_map(model, device_map, offload_dir=None)`

Dispatches a model according to an explicit `DeviceMap` — a dict mapping module name to `(onload_device, offload_device)` tuples.

```python
device_map = {
    "model.embed_tokens":  ("cuda:0", "cuda:0"),
    "model.layers.0":      ("cuda:0", "cpu"),
    "model.layers.1":      ("cuda:0", "disk"),
}
dispatch_with_map(model, device_map, offload_dir="/tmp/offload")
```

**When to use:** when you have a specific, manually-determined dispatch plan. Also used internally by `from_accelerate`.

---

#### `get_device_map(model, default_device=cpu) → DeviceMap`

Introspects a dispatched model and returns a `DeviceMap` describing each module's current `(onload_device, offload_device)`.

```python
device_map = get_device_map(model)
# -> {"model.layers.0": (cuda:0, cpu), "model.layers.1": (cuda:0, disk), ...}
```

**Use when:** you need to inspect, serialize, or replicate the dispatch configuration of a model.

---

#### `remove_dispatch(module, onload_tensors=False) → module`

Removes all offloading from every submodule of `module`.

```python
remove_dispatch(model, onload_tensors=True)  # bring all weights to GPU before saving
```

---

#### `get_device_memory() → dict[torch.device, int]`

Returns a mapping of all available CUDA devices to their total memory (in bytes). In a distributed context, returns only the local rank's GPU. Returns an empty dict if no CUDA devices are available.

---

### `load.py` — Loading with Offloading

#### `load_offloaded_model(extra_cpu_mem=5e9)` *(context manager)*

A context manager that patches `from_pretrained` on any `transformers` model class or `PreTrainedModel` subclass visible in the caller's global scope. Within the context, calling `from_pretrained` will:

1. On rank 0: load the model using `accelerate`'s device offloading.
2. On other ranks: load the model on the meta device (no actual weights).
3. After loading: call `from_accelerate` to convert to `compressed_tensors` offloading and share weights across ranks.

```python
from transformers import AutoModelForCausalLM

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B",
        device_map="auto_offload",  # CT-specific: only cpu/disk, no GPU
    )
```

**Special `device_map` value: `"auto_offload"`**

In addition to standard `device_map` options (`"auto"`, `"cpu"`, etc.), `load_offloaded_model` supports `device_map="auto_offload"`. This restricts accelerate's auto-mapping to only use CPU and disk (no GPU VRAM), which is useful when you want the GPU to be fully available for activations during inference.

**Parameters:**
- `extra_cpu_mem`: bytes to reserve in CPU RAM for non-weight operations (default 5 GB). Memory estimates are automatically computed for both distributed and non-distributed setups.

**When to use:** the recommended entry point for loading offloaded models from pretrained checkpoints. Handles distributed loading correctly, sharing weights across ranks with minimal memory overhead.

---

### `convert/` — Accelerate Interoperability

The `convert` submodule provides bidirectional conversion between `compressed_tensors` offloading and HuggingFace `accelerate` offloading. This is needed because the `transformers` ecosystem (loading, saving, PEFT, etc.) expects accelerate's `AlignDevicesHook`-based offloading.

#### `from_accelerate(model) → (device_map, offload_dir)` — `convert/from_accelerate.py`

Converts a model from accelerate offloading to `compressed_tensors` offloading. Called automatically by `load_offloaded_model`.

Process:
1. Calls `remove_accelerate(model)` to strip accelerate hooks and collect device/offload information.
2. In a distributed context, broadcasts the `device_map` and `offload_dir` from rank 0 to all ranks.
3. Calls `dispatch_with_map` to apply `compressed_tensors` offloading.

For disk-offloaded modules, the accelerate disk index is copied into `DiskCache.index` so tensors can be read via `safe_open` without re-writing files.

**Returns:** `(device_map, offload_dir)` — the device map and optional disk offload directory.

---

#### `remove_accelerate(model) → (device_map, offload_dir)`

Strips all `AlignDevicesHook` instances from every module in the model. For each module, calls `remove_accelerate_from_module`. Removes `model.hf_device_map` if present.

---

#### `remove_accelerate_from_module(module) → (onload_device, offload_device, offload_dir)`

Removes the accelerate `AlignDevicesHook` from a single module. Handles three offload cases:

- **CPU/device offload**: tensors in `hook.weights_map` are reassigned directly.
- **Disk offload**: meta tensors are kept as-is; the accelerate disk index is copied into `DiskCache.index`.
- **No offload**: no-op; returns the module's current device for both onload and offload.

This is a zero-copy operation — no new tensors are allocated.

---

#### `to_accelerate(model) → hf_device_map` — `convert/to_accelerate.py`

Converts a `compressed_tensors`-offloaded model back to accelerate offloading. This is necessary before calling `model.save_pretrained()`, since `transformers`'s saving logic understands `AlignDevicesHook` but not `OffloadCache`.

```python
# Before saving:
to_accelerate(model)
model.save_pretrained("./output")

# After saving, convert back for inference:
from_accelerate(model)
```

For disk-offloaded modules, creates an `OffloadedWeightsLoader` backed by the `DiskCache` index. For CPU/device-offloaded modules, uses an in-memory dict as the weights map.

Sets `model.hf_device_map` as expected by `transformers`.

---

#### `to_accelerate_module(module, name=None, hf_disk_index=None) → str`

Lower-level version of `to_accelerate` that converts a single module. Returns the string representation of the module's offload device.

---

### `__init__.py` — Top-Level API

The top-level `compressed_tensors.offload` namespace re-exports the most important functions and adds several convenience utilities.

#### `disable_offloading()` *(context manager)*

Disables offloading globally. Within this context, onloaded tensors are cached so that repeated accesses do not trigger additional device-to-device copies. All cached values are released on exit.

```python
with disable_offloading():
    # All subsequent accesses to offloaded parameters are cache hits
    for token in sequence:
        output = model(token)
```

**Use when:** processing multiple tokens or batches without wanting to offload between each call.

---

#### `disable_onloading()` *(context manager)*

Disables onloading globally. Parameter reads return the raw offloaded tensors; assignments do not trigger offloading. Primarily used for debugging and internal utilities.

```python
with disable_onloading():
    raw_cpu_tensor = module.weight  # returns the CPU tensor
    module.weight = new_tensor       # assigns without triggering copy
```

---

#### `update_offload_parameter(module, name, data)`

Updates an offloaded parameter or buffer in-place. Works for both offloaded and non-offloaded modules.

For offloaded modules: calls the cache's `update_offload` (copies into shared CPU memory, overwrites safetensors file, etc.).
For non-offloaded modules: uses `param.copy_(data)`.

```python
update_offload_parameter(layer, "weight", quantized_weight)
```

> **Caution:** does not guard against multiple ranks writing simultaneously. The caller is responsible for ensuring single-rank writes. Also does not broadcast onloaded values to other ranks.

**Use when:** performing quantization, weight updates, or any post-load modification to model weights.

---

#### `get_execution_device(module, default=None) → torch.device | "disk"`

Returns the device that input tensors should be moved to before running the module's forward pass.

For offloaded modules: returns `module._parameters.onload_device`.
For non-offloaded modules: returns the device of the first parameter/buffer.

---

#### `get_offloaded_device(module, default=None) → torch.device | "disk"`

Returns the device where the module's weights reside when not in use (between forward passes).

For offloaded modules: returns `module._parameters.offload_device`.
For non-offloaded modules: returns the device of the first parameter/buffer.

---

#### `register_offload_module(base, name, module)`

Registers a new submodule on a parent module, propagating offloading if the parent is offloaded.

```python
# Attaches `new_layer` with the same offload config as `model.layers[0]`
register_offload_module(model, "new_layer", new_layer)
```

**Use when:** dynamically adding submodules to an already-dispatched model (e.g., adding LoRA adapters after dispatch).

---

#### `align_modules(modules, execution_device=None)` *(deprecated)*
#### `align_module_device(module, execution_device=None)` *(deprecated)*

Legacy context managers for temporarily moving a module's parameters to a specific device. Kept for backwards compatibility.

For offloaded modules: temporarily overrides `onload_device` and disables offloading.
For non-offloaded modules: moves tensors manually and restores them on exit.

> Use `disable_offloading()` combined with `get_execution_device()` for new code.

---

### `dist_utils.py` — Distributed Utilities

#### `is_distributed() → bool`

Returns `True` if `torch.distributed` is available and has been initialized.

#### `is_rank0() → bool`

Returns `True` if not in a distributed context, or if the current process is rank 0.

#### `init_dist()`

Initializes the distributed process group using NCCL and environment variables set by `torchrun`. Sets the CUDA device to the local rank. Requires the `TORCHELASTIC_RUN_ID` environment variable (set automatically by `torchrun`).

```bash
torchrun --nproc-per-node=4 my_script.py
```

```python
# At the top of my_script.py:
from compressed_tensors.offload import init_dist
init_dist()
```

#### `as_broadcastable(tensor) → tensor`

Returns a view of the tensor compatible with `dist.broadcast`. Works around an NCCL limitation: FP8 dtypes (`float8_e4m3fn`, etc.) cannot be broadcast on pre-Hopper hardware. FP8 tensors are reinterpreted as `uint8` for the broadcast and share the same underlying storage.

---

### `utils.py` — Internal Utilities

These utilities are used internally throughout the module.

#### `send_tensors(value, *args, **kwargs) → value`

Recursively traverses a nested structure (tensor, list, tuple, dict, or dataclass) and calls `.to(*args, **kwargs)` on all tensors found. Returns a new structure with the moved tensors, preserving the original's type and `__dict__`.

Used by the forward wrapper to move input tensors to the execution device.

#### `get_module_device(module, default=None) → torch.device`

Returns the device of the first parameter or buffer of a module. Falls back to `default` if the module has no tensors.

#### `move_module_tensor(module, name, device)`

Moves a single named parameter or buffer to a new device.

#### `module_size(module, recurse=True) → int`

Returns the total size in bytes of all parameters and buffers in a module. Uses `disable_onloading` to avoid triggering device movement.

#### `get_module_sizes(model, no_split_modules=()) → list[(module, int)]`

Returns a flat list of `(module, size_bytes)` for all non-splittable modules in the model. A module is considered non-splittable if it has direct parameters or if its class name is in `no_split_modules`. Used by `dispatch_model` to pack modules onto devices.

#### `to_empty(tensor, **kwargs) → tensor`

Creates an empty tensor with the same shape, dtype, and subclass as the input tensor. Equivalent to `torch.empty_like` but preserves tensor subclass and `__dict__`.

#### `to_tensor(dst, src) → dst`

Copies the subclass, `__dict__`, and `requires_grad` from `src` into `dst`. Used during accelerate conversion to "convert" an accelerate tensor into the corresponding `compressed_tensors` tensor without copying data.

---

## Common Usage Patterns

### 1. Loading a Large Model for Inference

```python
from transformers import AutoModelForCausalLM
from compressed_tensors.offload import load_offloaded_model

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-70B",
        device_map="auto",      # or "auto_offload" to restrict to cpu/disk only
        torch_dtype="bfloat16",
    )
# model is now using compressed-tensors offloading
```

### 2. Multi-GPU Dispatch

```python
from compressed_tensors.offload import dispatch_model

model = ...  # load on CPU or meta device
model = dispatch_model(model)  # automatically fills available GPUs, offloads remainder to CPU
```

### 3. Distributed Inference with torchrun

```python
# run with: torchrun --nproc-per-node=4 script.py
from transformers import AutoModelForCausalLM
from compressed_tensors.offload import init_dist, load_offloaded_model

init_dist()  # initialize NCCL

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-70B",
        device_map="auto",
        torch_dtype="bfloat16",
    )
# rank 0 loads weights; other ranks get weights via broadcast
# all ranks share CPU memory via /dev/shm
```

### 4. Updating Weights After Loading (e.g., Post-Quantization)

```python
from compressed_tensors.offload import update_offload_parameter

for name, module in model.named_modules():
    if hasattr(module, "weight"):
        quantized = quantize(module.weight)
        update_offload_parameter(module, "weight", quantized)
```

### 5. Saving an Offloaded Model

```python
from compressed_tensors.offload import to_accelerate, from_accelerate

# Convert to accelerate for saving
to_accelerate(model)
model.save_pretrained("./output")

# Convert back for continued inference
from_accelerate(model)
```

### 6. Manual Dispatch with a Device Map

```python
from compressed_tensors.offload import dispatch_with_map

device_map = {
    "model.embed_tokens": ("cuda:0", "cuda:0"),
    "model.layers.0":     ("cuda:0", "cpu"),
    "model.layers.1":     ("cuda:0", "disk"),
    # ...
}
dispatch_with_map(model, device_map, offload_dir="/tmp/weights")
```

### 7. Inspecting Dispatch Configuration

```python
from compressed_tensors.offload import get_device_map, get_execution_device, get_offloaded_device

device_map = get_device_map(model)
exec_dev = get_execution_device(model.layers[0])   # cuda:0
offload_dev = get_offloaded_device(model.layers[0]) # cpu
```

---

## Architecture Diagram

```
compressed_tensors.offload
├── load.py                   load_offloaded_model()
│     └── calls from_accelerate after loading
│
├── dispatch.py               dispatch_model(), set_onload_device(), dispatch_with_map()
│     └── calls offload_module() for each module
│
├── module.py                 offload_module(), remove_module_offload()
│     └── replaces _parameters/_buffers with OffloadCache
│         wraps module.forward to move inputs
│
├── cache/
│   ├── base.py              OffloadCache (abstract MutableMapping)
│   ├── cpu.py               CPUCache
│   ├── device.py            DeviceCache
│   ├── disk.py              DiskCache
│   ├── dist_cpu.py          DistributedCPUCache (shared memory)
│   ├── dist_device.py       DistributedDeviceCache (broadcast)
│   └── dist_disk.py         DistributedDiskCache (shared files)
│
├── convert/
│   ├── from_accelerate.py   from_accelerate() — converts from HF accelerate
│   ├── to_accelerate.py     to_accelerate()   — converts to HF accelerate
│   └── helpers.py           norm_device(), get_tensors()
│
├── dist_utils.py            is_distributed(), init_dist(), as_broadcastable()
├── utils.py                 send_tensors(), module_size(), get_module_sizes()
└── __init__.py              public API + context managers + parameter utils
```
