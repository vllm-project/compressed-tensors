# LUT-B DP final-endpoint benchmark

Weighted 1D DP with 512 values and 8 centers per tile, run on an H100 using
CUDA events. Inputs use independent random values and positive nonuniform
weights. The timings include sorting, prefix construction, and the fused DP
kernel. The endpoint-only variant computes complete DP rows 1–7, then only
`f(512, 8)`. Candidate block size 32 is held fixed for the direct comparison.

| 512-point tiles | Full final row | Only `f(512, 8)` | Speedup | Center difference |
|---:|---:|---:|---:|---:|
| 512 | 1.966 ms | 1.814 ms | 1.08× | 0 |
| 4,096 | 4.795 ms | 4.276 ms | 1.12× | 0 |
| 16,384 | 16.552 ms | 14.678 ms | 1.13× | 0 |

Scanning the endpoint candidates in blocks larger than 32 did not help at the
larger batch sizes. At 512 tiles, block 64 was slightly faster, but that
measurement also changes the block size used in the earlier DP rows, so the
matched block-32 comparison above is the clean result.

The endpoint pruning is faster, but the full-fit gain is about 8–11% in these
measurements because sorting, prefix construction, and rows 1–7 are unchanged.
The benchmark driver is `benchmark_lut_b_dp_final_endpoint.py`; raw measurements
are in `lutb_dp_final_endpoint_benchmark.json`.
