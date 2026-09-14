import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
rows = []
for p in sorted(args.root.glob('*/**/trace_view.json')):
    data = json.loads(p.read_text())
    events = data.get('traceEvents', []) if isinstance(data, dict) else data
    complete = [x for x in events if x.get('ph') == 'X']
    for x in complete:
        x['ts'] = float(x['ts'])
        x['dur'] = float(x.get('dur', 0))
    markers = [x for x in complete if 'step[TARGET_VERIFY bs=162]' in x.get('name', '')]
    selected = complete
    if markers:
        selected = [x for x in complete if any(x['ts'] >= m['ts'] and x['ts'] + x.get('dur', 0) <= m['ts'] + m['dur'] for m in markers)]
    names = Counter(x['name'] for x in selected if 'FusedInferAttention' in x.get('name', ''))
    metrics = {}
    for name, count in names.items():
        dur = [x['dur'] for x in selected if x['name'] == name]
        metrics[name] = {'count': count, 'sum_us': sum(dur), 'mean_us': statistics.mean(dur)}
    row = {'trace': str(p), 'steps': len(markers), 'step_mean_us': statistics.mean([m['dur'] for m in markers]) if markers else None, 'step_p50_us': statistics.median([m['dur'] for m in markers]) if markers else None, 'ifa': metrics}
    rows.append(row)
    print(json.dumps(row))
(args.root / 'trace_summary.json').write_text(json.dumps(rows, indent=2))
