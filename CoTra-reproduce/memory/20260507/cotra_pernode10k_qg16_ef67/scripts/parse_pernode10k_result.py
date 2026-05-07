#!/usr/bin/env python3
import re, sys
from pathlib import Path
case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_pernode10k_qg16_ef67')
logs = case / 'logs'
results = case / 'results'
line_re = re.compile(r'^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(?:us\s+)?([0-9.]+)\s*(?:/s)?')
rows = []
for lf in sorted(logs.glob('anns_q*_qg*_job*_*_0.log')):
    text = lf.read_text(errors='replace').splitlines()
    q = ''
    qg = ''
    per_node = ''
    for line in text:
        if 'QUERY_COUNT=' in line:
            m = re.search(r'QUERY_COUNT=(\d+)', line)
            if m: q = int(m.group(1)); per_node = q // 8
            m = re.search(r'COTRA_QUERY_GROUP_SIZE=(\d+)', line)
            if m: qg = int(m.group(1))
    parsed = []
    for line in text:
        m = line_re.match(line)
        if not m:
            continue
        ef = int(m.group(1)); recall = float(m.group(2)); avg = float(m.group(3)); qps = float(m.group(4))
        if 0 < ef <= 5000 and 0.0 <= recall <= 1.0:
            parsed.append((ef, recall, avg, qps, line.strip()))
    measure = [x for x in parsed if x[0] == 67]
    if measure:
        ef, recall, avg, qps, raw = measure[-1]
        rows.append((q, per_node, qg, ef, recall, avg, qps, 'OK', str(lf), raw))
    else:
        rows.append((q, per_node, qg, '', '', '', '', 'NO_RESULT', str(lf), ''))
out = results / 'pernode10k_qg16_summary.tsv'
with out.open('w') as f:
    f.write('query_count\tper_node_query_count\tquery_group_size\tef\trecall\tavg_lat_reported\tqps\tstatus\tleader_log\traw\n')
    for row in rows:
        f.write('\t'.join(map(str, row)) + '\n')
print(out)
