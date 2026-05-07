# Error / Issue Log

- No fatal benchmark error: Slurm job `9722160` completed with exit code `0:0`; `logs/slurm-9722160.err` is empty; application status printed `ANN_STATUS=0`.
- Each rank log starts with `error connecting to /tmp/tmux-204881/default (No such file or directory)`. This did not stop the benchmark and appears before normal CoTra output; it is recorded as a non-fatal environment message.
- A separate older q-sweep job `9722161` completed while the global CoTra binary was temporarily rebuilt with `QUERY_GROUP_SIZE=16`. That job is not used as a QG4 baseline here; its sacct snapshot is saved as `logs/sacct_9722161_not_this_case.txt` to prevent accidental reuse.
- To avoid future contamination, `include/anns/anns_config.h` and the main `build/tests/scala_anns` binary were restored/rebuilt to `QUERY_GROUP_SIZE=4` after saving the QG16 binary snapshot.

- Post-processing note: one artifact-listing shell command exited nonzero because of quoting in the final ls command; the files were subsequently inspected and verified. No benchmark artifact was lost.
