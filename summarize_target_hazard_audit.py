"""Compact audit progress and results, including actual gate engagements."""
import json
from pathlib import Path
from collections import Counter
root=Path('artifacts/target_hazard_gate_20260914/audit_v2')
for path in sorted(root.glob('*/results.json')):
    rows=json.loads(path.read_text())['episodes']
    for stratum in ('all','unfiltered','sand_crossing'):
        selected=[r for r in rows if stratum=='all' or r['stratum']==stratum]
        ticks=sum(len(r['teacher_ticks']) for r in selected)
        reasons=Counter()
        for r in selected:
            reasons.update(r['gate']['engagements'])
        print(json.dumps(dict(controller=path.parent.name,stratum=stratum,n=len(selected),
            successes=sum(r['success'] for r in selected),falls=sum(r['fall'] for r in selected),
            abstentions=sum(r['planning_failed'] for r in selected),timeouts=sum(r['timeout'] for r in selected),
            sand_seconds=sum(r['sand_seconds'] for r in selected),
            takeover=sum(sum(r['teacher_ticks']) for r in selected)/max(1,ticks),
            teacher_cost=sum(r['teacher_executed_cost'] for r in selected),
            student_cost=sum(r['student_executed_cost'] for r in selected),
            planning_failures=sum(r['gate']['planning_failures'] for r in selected),engagements=dict(reasons))))
