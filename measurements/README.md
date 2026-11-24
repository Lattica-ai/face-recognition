# Measurements

`harness/run_submission.py` writes one JSON measurement file per run into a
sub-directory named for the instance size: `single`, `small`, `medium`, or
`large`.

Running with `--num_runs <n>` produces `results-1.json` … `results-<n>.json` in
the corresponding sub-directory.

## Submitting

The submission on this branch serves the single-pair instance only. This command
assumes the repository's `.venv` is active as described in the top-level README:

```console
python3 harness/run_submission.py 0 --num_runs 3   # 1 pair per run
```

## File schema

Each `results-*.json` follows the FHE-benchmarking measurement schema:

- **`Timing`** — wall-clock latency per stage (e.g. `Key Generation`,
  `Encrypted model preprocessing`, `Input encryption`, `Encrypted computation`,
  `Result decryption`, …), plus `Total`.
- **`Bandwidth`** — sizes of `Public and evaluation keys`, `Packed model
  weights`, `Encrypted input`, and `Encrypted results`.
- **`Quality`** — for single inference, the encrypted similarity `score` and
  ground-truth `label`; for batched inference, `Encrypted model quality` and
  `Harness model quality`, each reporting `eer`, `tar_at_far_1pct`, and
  `tar_at_far_01pct`.
- **`Server Reported`** — the remote worker's own timing of the query: the pure
  `Encrypted computation` on the GPU, the backend's `Total`, and its overhead
  and upload shares.

The `Quality` block also contains the encrypted-minus-ArcFace metric gaps. A
batched run passes the quality criterion when its encrypted EER is no more than
0.15 above the ArcFace EER on the same sampled pairs.

Published workload pages and result tables live in the separate
[`fhe-benchmarking.github.io`](https://github.com/fhe-benchmarking/fhe-benchmarking.github.io/tree/main/face-recognition)
repository.
