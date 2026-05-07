#!/usr/bin/env python3
import csv
import glob
import math
import re
import statistics
import sys
from pathlib import Path

case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_qscale_mem_sift100m')
logs = case / 'logs'
results = case / 'results'
mem_root = case / 'mem_samples'
results.mkdir(parents=True, exist_ok=True)

line_re = re.compile(r'^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(?:us\s+)?([0-9.]+)\s*(?:/s)?')
log_re = re.compile(r'anns_q(\d+)_qg(\d+)_job(\d+)_([^_]+)_(\d+)\.log$')
actual_mem_re = re.compile(r'Actual memory usage:\s*([0-9.]+)\s*Mb')
config_re = re.compile(r'COTRA_SEARCH_ONLY_CONFIG\s+ef=(\d+)\s+skip_recall=(\d+)\s+query_group_size=(\d+)\s+query_load_per_rank=(\d+)\s+qsize=(\d+)')

def fmean(vals):
    return statistics.mean(vals) if vals else float('nan')

def fmedian(vals):
    return statistics.median(vals) if vals else float('nan')

def fstdev(vals):
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0

def trimmed_mean_drop_min_max(vals):
    if len(vals) >= 3:
        ordered = sorted(vals)
        return statistics.mean(ordered[1:-1])
    return fmean(vals)

def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    if isinstance(x, float):
        return f'{x:.{digits}f}'
    return str(x)

# Status rows
status_by_job = {}
for p in sorted(results.glob('status_*.tsv')):
    with p.open(newline='') as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for row in rdr:
            if row.get('job_id'):
                status_by_job[row['job_id']] = row

per_rank_rows = []
for lf in sorted(logs.glob('anns_q*_qg*_job*_*_*.log')):
    m = log_re.match(lf.name)
    if not m:
        continue
    q = int(m.group(1)); qg = int(m.group(2)); job = m.group(3); node = m.group(4); rank = int(m.group(5))
    parsed = []
    configs = []
    actual_mb = None
    with lf.open(errors='replace') as f:
        for line in f:
            cm = config_re.search(line)
            if cm:
                configs.append({
                    'ef': int(cm.group(1)),
                    'skip_recall': int(cm.group(2)),
                    'query_group_size': int(cm.group(3)),
                    'query_load_per_rank': int(cm.group(4)),
                    'qsize': int(cm.group(5)),
                })
            mm = line_re.match(line)
            if mm:
                ef = int(mm.group(1)); recall = float(mm.group(2)); avg = float(mm.group(3)); qps = float(mm.group(4))
                if 0 < ef <= 5000 and 0.0 <= recall <= 1.0:
                    parsed.append((ef, recall, avg, qps, line.strip()))
            am = actual_mem_re.search(line)
            if am:
                actual_mb = float(am.group(1))
    for idx, (ef, recall, avg, qps, raw) in enumerate(parsed):
        per_rank_rows.append({
            'job_id': job, 'query_count': q, 'per_node_query_count': q // 8, 'query_group_size': qg,
            'node': node, 'rank': rank, 'run_index_in_log': idx, 'ef': ef, 'recall': recall,
            'avg_or_time_reported': avg, 'qps': qps, 'actual_memory_mb': actual_mb,
            'leader': 1 if rank == 0 else 0, 'log': str(lf), 'raw': raw,
        })

per_rank_out = results / 'per_rank_all_ef_runs.tsv'
with per_rank_out.open('w', newline='') as f:
    cols = ['job_id','query_count','per_node_query_count','query_group_size','node','rank','leader','run_index_in_log','ef','recall','avg_or_time_reported','qps','actual_memory_mb','log','raw']
    wr = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
    wr.writeheader(); wr.writerows(per_rank_rows)

leader_rows = [r for r in per_rank_rows if r['leader'] == 1 and r['ef'] == 67]
leader_out = results / 'leader_ef67_runs.tsv'
with leader_out.open('w', newline='') as f:
    cols = ['job_id','query_count','per_node_query_count','query_group_size','run_index_in_log','ef','recall','qps','actual_memory_mb','log','raw']
    wr = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
    wr.writeheader()
    for r in leader_rows:
        wr.writerow({c: r.get(c, '') for c in cols})

summary_rows = []
for q in sorted({r['query_count'] for r in leader_rows}):
    rows = [r for r in leader_rows if r['query_count'] == q]
    qps = [r['qps'] for r in rows]
    rec = [r['recall'] for r in rows]
    jobs = sorted({r['job_id'] for r in rows})
    qg = rows[0]['query_group_size'] if rows else ''
    status = ';'.join(f'{j}:{status_by_job.get(j, {}).get("status", "")}' for j in jobs)
    runtime = ';'.join(f'{j}:{status_by_job.get(j, {}).get("runtime_sec", "")}' for j in jobs)
    summary_rows.append({
        'query_count': q, 'per_node_query_count': q // 8, 'query_group_size': qg, 'ef': 67,
        'runs': len(rows), 'recall_mean': fmean(rec), 'recall_median': fmedian(rec),
        # Report qps_mean using the agreed 5-run trimmed mean: drop max/min,
        # then average the remaining 3 values. Keep arithmetic mean for audit.
        'qps_mean': trimmed_mean_drop_min_max(qps), 'qps_arithmetic_mean': fmean(qps),
        'qps_median': fmedian(qps), 'qps_min': min(qps) if qps else float('nan'),
        'qps_max': max(qps) if qps else float('nan'), 'qps_stdev': fstdev(qps),
        'jobs': ','.join(jobs), 'job_status': status, 'runtime_sec': runtime,
    })
summary_out = results / 'qps_summary.tsv'
with summary_out.open('w') as f:
    cols = ['query_count','per_node_query_count','query_group_size','ef','runs','recall_mean','recall_median','qps_mean','qps_arithmetic_mean','qps_median','qps_min','qps_max','qps_stdev','jobs','job_status','runtime_sec']
    f.write('\t'.join(cols) + '\n')
    for r in summary_rows:
        f.write('\t'.join(fmt(r[c], 6 if c.startswith('recall') else 3) for c in cols) + '\n')

# Memory samples
per_node_mem = []
for d in sorted(mem_root.glob('q*_job*')):
    dm = re.match(r'q(\d+)_job(\d+)$', d.name)
    if not dm:
        continue
    q = int(dm.group(1)); job = dm.group(2)
    for mf in sorted(d.glob('*.tsv')):
        # Name is cpu1-90_rank0.tsv.
        nm = re.match(r'(.+)_rank(\d+)\.tsv$', mf.name)
        node = nm.group(1) if nm else ''
        rank = int(nm.group(2)) if nm else -1
        max_rss = max_hwm = max_vsz = max_sys_used = 0
        first_ts = last_ts = ''
        samples = 0
        with mf.open(newline='') as f:
            rdr = csv.DictReader(f, delimiter='\t')
            for row in rdr:
                samples += 1
                if not first_ts:
                    first_ts = row.get('timestamp', '')
                last_ts = row.get('timestamp', '')
                def geti(key):
                    try:
                        return int(float(row.get(key, '0') or 0))
                    except ValueError:
                        return 0
                max_rss = max(max_rss, geti('vmrss_kb'))
                max_hwm = max(max_hwm, geti('vmhwm_kb'))
                max_vsz = max(max_vsz, geti('vmsize_kb'))
                max_sys_used = max(max_sys_used, geti('system_used_kb'))
        # Add program-reported Actual memory usage if present.
        actual_mb = None
        log_matches = list(logs.glob(f'anns_q{q}_qg*_job{job}_{node}_{rank}.log'))
        if log_matches:
            text = log_matches[0].read_text(errors='replace')
            mm = actual_mem_re.search(text)
            if mm:
                actual_mb = float(mm.group(1))
        per_node_mem.append({
            'job_id': job, 'query_count': q, 'per_node_query_count': q // 8, 'node': node, 'rank': rank,
            'samples': samples, 'first_ts': first_ts, 'last_ts': last_ts,
            'max_vmrss_kb': max_rss, 'max_vmhwm_kb': max_hwm, 'max_vmsize_kb': max_vsz,
            'max_system_used_kb': max_sys_used,
            'max_vmrss_gib': max_rss / 1024 / 1024,
            'max_vmhwm_gib': max_hwm / 1024 / 1024,
            'max_vmsize_gib': max_vsz / 1024 / 1024,
            'max_system_used_gib': max_sys_used / 1024 / 1024,
            'actual_memory_mb': actual_mb if actual_mb is not None else '',
            'actual_memory_gib': actual_mb / 1024 if actual_mb is not None else '',
            'mem_file': str(mf),
        })

per_node_out = results / 'per_node_memory.tsv'
with per_node_out.open('w', newline='') as f:
    cols = ['job_id','query_count','per_node_query_count','node','rank','samples','first_ts','last_ts','max_vmrss_kb','max_vmhwm_kb','max_vmsize_kb','max_system_used_kb','max_vmrss_gib','max_vmhwm_gib','max_vmsize_gib','max_system_used_gib','actual_memory_mb','actual_memory_gib','mem_file']
    wr = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
    wr.writeheader()
    for r in per_node_mem:
        wr.writerow(r)

mem_summary_rows = []
for q in sorted({r['query_count'] for r in per_node_mem}):
    rows = [r for r in per_node_mem if r['query_count'] == q]
    rss = [r['max_vmrss_gib'] for r in rows]
    hwm = [r['max_vmhwm_gib'] for r in rows]
    actual = [r['actual_memory_gib'] for r in rows if isinstance(r['actual_memory_gib'], float)]
    sysused = [r['max_system_used_gib'] for r in rows]
    jobs = sorted({str(r['job_id']) for r in rows})
    mem_summary_rows.append({
        'query_count': q, 'per_node_query_count': q // 8, 'nodes': len(rows), 'jobs': ','.join(jobs),
        'proc_rss_gib_mean': fmean(rss), 'proc_rss_gib_median': fmedian(rss), 'proc_rss_gib_min': min(rss) if rss else float('nan'), 'proc_rss_gib_max': max(rss) if rss else float('nan'),
        'proc_hwm_gib_mean': fmean(hwm), 'proc_hwm_gib_max': max(hwm) if hwm else float('nan'),
        'program_actual_gib_mean': fmean(actual), 'program_actual_gib_max': max(actual) if actual else float('nan'),
        'system_used_gib_mean': fmean(sysused), 'system_used_gib_max': max(sysused) if sysused else float('nan'),
    })
mem_summary_out = results / 'memory_summary.tsv'
with mem_summary_out.open('w') as f:
    cols = ['query_count','per_node_query_count','nodes','jobs','proc_rss_gib_mean','proc_rss_gib_median','proc_rss_gib_min','proc_rss_gib_max','proc_hwm_gib_mean','proc_hwm_gib_max','program_actual_gib_mean','program_actual_gib_max','system_used_gib_mean','system_used_gib_max']
    f.write('\t'.join(cols) + '\n')
    for r in mem_summary_rows:
        f.write('\t'.join(fmt(r[c], 3) for c in cols) + '\n')

print(f'wrote {per_rank_out}')
print(f'wrote {leader_out}')
print(f'wrote {summary_out}')
print(f'wrote {per_node_out}')
print(f'wrote {mem_summary_out}')
