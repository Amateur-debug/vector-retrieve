# Error log

- CoTra job `9722148` (`-Q 640000`, amplified query/GT) timed out after `01:00:27`; no EF result line was printed. Rank logs reached `query size: 640000 vecsize: 128` and then stayed CPU-bound. Treat this as no valid QPS for the true one-shot qsize=640K path.
- CoTra job `9722151` was an invalid fallback submission: it did not exclude the already-running q640K nodes, so Slurm placed it on `cpu1-[30-37]`; `memcached` failed to listen on `10.120.30.30:24670` with `Address already in use`. No measurement was produced.
- CoTra fallback job `9722153` used `--exclude=cpu1-[30-37]` and `MEM_PORT=24671`; it completed successfully.
- XY job `9722149` completed successfully. Sacct shows some worker `srun` steps as `CANCELLED` at cleanup because the script intentionally kills long-lived `node_start.py` servers after query phases; the overall job and query phases completed with status 0.
- The repeated CoTra fallback is recorded as an approximation, not a fully fair equivalent of XY's 64 concurrent senders.
