# CoTra original-vs-current implementation comparison

Date: 2026-05-07 Asia/Shanghai
Current repo: `/hpc2hdd/home/rmeng603/workspace/CoTra`
Original repo: `/hpc2hdd/home/rmeng603/workspace/vector-retrieve/CoTra/CoTra`

## Goal

Compare the original implementation with the current runnable implementation, and judge whether the current changes introduced the QPS / large-batch problems being observed.

## Method

Static comparison only; no new build or Slurm performance job was submitted. The original code is known not to run in the current environment because it contains hard-coded compiler and third-party library paths, so the first pass avoids using it as an executable baseline.

Commands used are recorded in `commands_used.txt`.

## Source identity

Both trees are at git commit `b2f4954`. The original tree is clean. The current tree contains local modifications and memory/build artifacts.

Important changed source files excluding build/memory:

- `CMakeLists.txt`, `third_party/CMakeLists.txt`
- `coromem/include/hwtopo.h`, `coromem/src/hwtopo.cpp`, `coromem/src/threadpool.cpp`
- `include/anns/anns_param.h`, `include/anns/exec_query.h`, `include/anns/scala_search.h`
- `include/index/scala_graph.h`
- `include/rdma/rdma_comm.h`, `include/rdma/rdma_config.h`, `src/rdma/rdma_comm.cc`
- `src/index/index_param.cc`
- `tests/scala_anns.cc`, `tests/anns/CMakeLists.txt`

## Findings

### Changes that are necessary or likely benign

- Build portability changes replace hard-coded compiler/spack paths with discoverable libraries. This is why current code can build/run in this environment.
- `COTRA_MACHINE_NUM` makes `MACHINE_NUM` configurable at CMake time instead of fixed only in `rdma_config.h`.
- Slurm/cpuset thread topology fallback and safe thread joins make runtime more robust.
- `scala_graph.h` fixes an allocation bug for `cumu_partition_num`: original allocated `sizeof(uint32_t) * partition_num + 1`, current allocates `sizeof(uint32_t) * (partition_num + 1)`.
- `src/rdma/rdma_comm.cc` fixes `wr_id` encoding by casting `query_id` to `uint64_t` before shifting. This is required for `qid >= 65536`; original code would overflow and can hang large batches.
- `COTRA_EF_LIST`, `COTRA_SKIP_RECALL`, and zero-division guards are test-control additions, not normal search-path changes unless the env vars are set.

### Shared original/current limitations

- `ScalaANN_v3` still splits query load as `qsize / MACHINE_NUM`.
- `scala_search_init()` still constructs `QueryMsg` for every global `qsize` on every rank, not just the local shard.
- `QueryMsg` still allocates/copies a query vector in the constructor, and its destructor is empty, so vector memory ownership is not cleaned up.

These are not newly introduced by the current runnable changes, but they explain why using very large `-Q` is not equivalent to a streaming workload.

### Suspicious current-only regression

The most suspicious current-only change is query-result lifecycle:

- Original `include/anns/exec_query.h` deletes each `QueryMsg* res` inside `proc_res()` immediately after recall processing.
- Current code removed that per-result `delete res` and instead deletes all `appr_alg.global_query[q]` only after the whole ef round finishes.

Impact: for large `qsize`, completed query objects and their internal queues/vectors/priority queues stay alive until the end of the round. This can amplify memory pressure and cache pressure, and is a plausible contributor to the Q=80K / per-node-10K QPS drop. It does not explain the original large-batch hang; that hang is better explained by the original `wr_id` overflow.

Caveat: both original and current `QueryMsg::~QueryMsg()` are empty, so neither version correctly frees the `vector` malloc. Original per-result delete still frees C++ container payloads earlier, while current keeps them alive longer.

### Possible low-priority performance differences

- Current CMake uses explicit `-msse2 -mavx2 -mfma`; original used `-march=native`. If the build host supports wider ISA, original could compile a slightly different distance kernel. Prior microbench showed SIMD arithmetic is not the main bottleneck, so this is lower priority.
- Current `checked_scala_v3_ngh_ptr()` adds bounds checks in hot graph traversal paths. It is useful for finding corrupt offsets and has not prevented current 10K runs from reaching ~83K QPS, but it may have a small overhead.

## Conclusion

The current runnable changes are not a blanket cause of poor QPS: the current tree already reproduced the integrated 10K path at about 83K QPS after warmup. However, the current removal of immediate `delete res` is a likely current-only regression for large `qsize` tests, because it retains all completed `QueryMsg` state until the end of the ef round. The full-global-`qsize` initialization problem existed in the original too, but the current lifecycle change makes it worse.

For the current problem, prioritize fixing query-state lifecycle/pooling before further increasing batch size. Keep the `wr_id` 64-bit fix; reverting it would reintroduce large-batch hangs.

## Recommended next validation

1. Restore early release semantics safely: after `proc_res()` finishes a query, clear or recycle its heavy containers immediately, but avoid dangling pointers and double frees.
2. Add a real `QueryMsg` destructor or explicit `release()` that frees `vector` and clears queues/vectors.
3. Run an A/B test on the current runnable tree: current lifecycle vs early-release lifecycle, with Q=10K and Q=80K, QG=16, ef=66 warmup then ef=67.
4. Keep `src/rdma/rdma_comm.cc` `wr_id` fix in both variants; otherwise Q>=65536 is not a valid comparison.

## Errors

No command error affected the analysis. One shell `printf` invocation treated a leading `---` string as an option; it only affected a section label and did not affect the diff output.
