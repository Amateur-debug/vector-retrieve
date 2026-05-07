#!/usr/bin/env python3
import glob, os, re, statistics, sys
from pathlib import Path

case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_batch_size_qsweep_ef67')
logs = case / 'logs'
results = case / 'results'
results.mkdir(parents=True, exist_ok=True)
status_rows = []
for p in sorted(results.glob('status_*.tsv')):
    with p.open() as f:
        header = next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 5:
                status_rows.append({'timestamp': parts[0], 'q': int(parts[1]), 'status': parts[2], 'runtime_sec': parts[3], 'log_glob': parts[4]})

line_re = re.compile(r'^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(?:us\s+)?([0-9.]+)\s*(?:/s)?')
rows = []
for lf in sorted(logs.glob('anns_q*_job*_*_0.log')):
    m = re.search(r'anns_q(\d+)_job', lf.name)
    if not m:
        continue
    q = int(m.group(1))
    text = lf.read_text(errors='replace').splitlines()
    is_leader = any('This is leader machine 0' in line for line in text) or lf.name.endswith('_0.log')
    parsed = []
    for line in text:
        mm = line_re.match(line)
        if mm:
            ef = int(mm.group(1)); recall = float(mm.group(2)); avg_or_time = float(mm.group(3)); qps = float(mm.group(4))
            # Ignore numeric data lines such as partition-size vectors; ef/recall lines have ef in the search range and recall in [0, 1].
            if 0 < ef <= 5000 and 0.0 <= recall <= 1.0:
                parsed.append((ef, recall, avg_or_time, qps, line.strip()))
    if not parsed:
        rows.append({'q': q, 'per_rank_q': q//8, 'ef': '', 'recall': '', 'avg_or_time_us': '', 'qps': '', 'status': 'NO_RESULT', 'leader_log': str(lf), 'raw': ''})
        continue
    measure = [x for x in parsed if x[0] == 67]
    ef, recall, avg_or_time, qps, raw = (measure[-1] if measure else parsed[-1])
    rows.append({'q': q, 'per_rank_q': q//8, 'ef': ef, 'recall': recall, 'avg_or_time_us': avg_or_time, 'qps': qps, 'status': 'OK', 'leader_log': str(lf), 'raw': raw})

# Attach Slurm/timeout status where available.
status_by_q = {}
for s in status_rows:
    status_by_q[s['q']] = s
for r in rows:
    s = status_by_q.get(r['q'])
    if s:
        r['job_status'] = s['status']
        r['runtime_sec'] = s['runtime_sec']
    else:
        r['job_status'] = ''
        r['runtime_sec'] = ''

rows.sort(key=lambda r: r['q'])
out = results / 'batch_size_summary.tsv'
with out.open('w') as f:
    cols = ['q','per_rank_q','ef','recall','avg_or_time_us','qps','status','job_status','runtime_sec','leader_log','raw']
    f.write('\t'.join(cols)+'\n')
    for r in rows:
        f.write('\t'.join(str(r.get(c,'')) for c in cols)+'\n')

# Basic derived statistics for repeated 10K baseline, if previous memory is present.
prev = Path('/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_integrated_640k_query_count_ef67/logs/anns_job9722153_cpu1-90_0.log')
prev_qps = []
if prev.exists():
    for line in prev.read_text(errors='replace').splitlines():
        mm = line_re.match(line)
        if mm and int(mm.group(1)) == 67:
            prev_qps.append(float(mm.group(4)))
prev_out = results / 'previous_repeated10k_qps.tsv'
with prev_out.open('w') as f:
    f.write('source\tn\tmean_qps\tmedian_qps\tmin_qps\tmax_qps\n')
    if prev_qps:
        f.write(f'{prev}\t{len(prev_qps)}\t{statistics.mean(prev_qps):.3f}\t{statistics.median(prev_qps):.3f}\t{min(prev_qps):.3f}\t{max(prev_qps):.3f}\n')
print(f'wrote {out}')
print(f'wrote {prev_out}')
