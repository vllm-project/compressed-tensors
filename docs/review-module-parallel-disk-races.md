# Race-condition review: `replace_module_parallel` with disk offloading

Reviews `src/compressed_tensors/distributed/module_parallel.py::replace_module_parallel`
under the assumption that modules are disk-offloaded and **share a common offload
directory** (`DistributedDiskCache`).

## How the pieces interact (why disk offload is special)

On-disk files are named `ct_disk_cache_{rank}_{id(offloaded)}.safetensors`
(`offload/cache/disk.py::_get_ct_file_path`). At load time, `DistributedDiskCache.offload`
writes the file on the source rank and **broadcasts the source's `safetensors_file`
path**; every non-source rank stores that same string in its class-level `index`
(`offload/cache/dist_disk.py`). So "a common disk offload" means many ranks hold, as a
bare path string, a reference to a file that only the source rank physically owns and
controls. That sharing-by-filename-string is where the hazards concentrate.

The 4-step algorithm:
1. **Decouple** — non-processing ranks `to_meta` their modules (under `disable_onloading`).
2. **Compress on meta** — non-processing ranks run `apply_fn` on meta to match the
   post-compression state dict (still under `disable_onloading`).
3. **Compress on device** — the processing rank runs `apply_fn` under `as_single_threaded`
   (so `DistributedDiskCache.offload` is patched to `DiskCache.offload`; writes are local,
   filenames still carry the true rank because `_get_rank` checks `dist.is_initialized()`
   directly rather than the patched `is_distributed`).
4. **Recouple** — for each module, `set_source_process(assigned_rank)` and re-`offload`
   each param/buffer, which broadcasts the authoritative disk path from the assigned rank
   to all others.

## Considered and dismissed: `id()`-reuse cross-rank corruption

Initial concern: filenames embed `id(offloaded)`, which is process-local and reusable
after GC, so the source could overwrite a file that a non-source rank still indexes by
string, yielding a silent wrong-tensor read.

**Not an issue.** The recouple broadcast in step 4 is authoritative. The source only
frees a meta tensor's `id` when it removes the entry from `index`, which happens **only**
in `DiskCache.__delitem__` with onloading enabled — i.e. only in step 3, and only for
weights assigned to the source. Every such weight is re-broadcast in step 4 with a fresh
path, so any non-source index that could have gone stale is overwritten with a correct
entry before it is ever read. Weights the source does *not* recompress keep their meta
tensors alive in `index` (step 2 runs under `disable_onloading`, which skips
`del self.index[...]`), so their `id`s are never freed and never reused. There is no
window where a non-source rank reads a reused-`id` file for a weight it still references.

## Genuine hazards

### 1. Shared-filesystem write visibility (no fsync, no step-3/step-4 barrier)

Step 3 writes files single-threaded on the processing rank; other ranks only record the
path (step 4) and read the bytes later at onload. `save_file` is synchronous but not
`fsync`'d, and there is **no barrier between step 3 and step 4** — the per-module
`dist.barrier()` inside `DistributedDiskCache.offload` orders the *metadata* exchange, not
the *data* write's visibility on a networked filesystem (NFS/Lustre). A later onload on a
different rank can observe a stale or partial file. Benign on a local FS; a real
cross-process race on the shared filesystems that motivate disk offload.

*Mitigation:* `fsync` after `save_file`, or a `dist.barrier()` at the end of step 3.

### 2. Read-after-delete for genuinely shared physical files (latent, guard-dependent)

Non-source ranks index the *source's* physical path for rewritten (non-symlink) files.
The source rank's `os.remove`/overwrite during compression is unsynchronized with any
concurrent reader. This is currently prevented in the common case by the step-2
`disable_onloading()` guard (it stops the source from deleting a shared file mid-compress
and stops non-source ranks from reading real data), so it is latent — but it depends
entirely on that guard remaining in place. Symlinked checkpoint files are safe regardless
(each rank has its own symlink name; `os.remove` drops only that link).

### 3. Dict-order coupling in the recouple broadcast (correctness assumption)

Step 4 zips `offloaded_values.items()` across ranks, but the per-tensor broadcast inside
`offload` carries no name key. Correct pairing relies on `apply_fn` inserting
params/buffers in the **same order** on the meta ranks (step 2) and the device rank
(step 3). If insertion order ever differs across ranks, weights are indexed to the wrong
files with no error. Works today only because `apply_fn` is assumed deterministic in
insertion order.

## Related fallout (not races)

The step-2 `disable_onloading()` guard means non-source ranks **never `os.remove` old
shared files and never drop their stale `index` entries**. This leaks disk files plus
meta-tensor keys in the class-level `index`, accumulating per layer. (Keeping those keys
alive is also what prevents `id` reuse for those tensors — the leak and the dismissed
corruption concern are two faces of the same lifecycle gap.)

## Priority

1. **Shared-FS write visibility** — add `fsync`/barrier before other ranks may read.
2. **Ordered file lifecycle / leak** — reconcile `index` and clean up old files on recouple.
3. **Dict-order assumption** — key the recouple broadcast by name, or assert consistent order.
