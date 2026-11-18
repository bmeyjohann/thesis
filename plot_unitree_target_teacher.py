"""Plot actual local-frame teacher trajectories against its authoritative cost grid."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
args=p.parse_args()
data=json.loads((args.directory/'results.json').read_text())
grid=np.load(args.directory/'planner_map.npz')
rows=data['episodes']
fig,axes=plt.subplots(math.ceil(len(rows)/4),4,figsize=(16,4*math.ceil(len(rows)/4)),squeeze=False)
for ax,row in zip(axes.flat,rows):
    cost=grid['cost'].copy()
    cost[grid['hard']]=30
    extent=[grid['axis'][0],grid['axis'][-1],grid['axis'][0],grid['axis'][-1]]
    ax.imshow(cost,origin='lower',extent=extent,cmap='YlOrBr',vmin=1,vmax=30)
    planned=row['planner']['path']
    if planned is not None:
        planned=np.asarray(planned);ax.plot(*planned.T,'--',color='white',lw=2,label='planned')
    trajectory=np.asarray(row['trajectory'])
    ax.plot(*trajectory.T,color='#167cad',lw=1.8,label='executed')
    if row.get('teacher_ticks'):
        from matplotlib.collections import LineCollection
        segments=np.stack([trajectory[:-1],trajectory[1:]],axis=1)
        takeover=np.asarray(row['teacher_ticks'][:len(segments)],dtype=bool)
        ax.add_collection(LineCollection(segments[takeover],colors='crimson',linewidths=2.2))
    ax.scatter(*row['start'],marker='o',c='lime',edgecolors='black',s=35)
    ax.scatter(*row['goal'],marker='*',c='red',edgecolors='black',s=90)
    if row['cost_points']:
        ax.scatter(*np.asarray(row['cost_points']).T,marker='x',color='red',s=80)
    state='success' if row['success'] else 'fall' if row['fall'] else row['planner']['status'] if row['planning_failed'] else 'timeout'
    ax.set_title(f"{row['seed']}: {state}, {row['seconds']:.1f}s, sand {row['sand_seconds']:.1f}s",fontsize=9)
    combined=np.vstack([trajectory,row['start'],row['goal']]+([planned] if planned is not None else []))
    lo=combined.min(0)-1;hi=combined.max(0)+1
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
for ax in axes.flat[len(rows):]:ax.set_visible(False)
controllers=sorted(set(Path(r['controller']).parent.parent.name+'/'+Path(r['controller']).stem if '/' in r['controller'] else r['controller'] for r in rows))
fig.suptitle(', '.join(controllers)+'\nWhite: reference plan / blue: novice / crimson: teacher / red crosses: falls; dark cells excluded')
fig.tight_layout()
fig.savefig(args.directory/'trajectories.png',dpi=140)
summary={}
for stratum in ['all']+sorted(set(r['stratum'] for r in rows)):
    selected=[r for r in rows if stratum=='all' or r['stratum']==stratum]
    summary[stratum]={key:sum(bool(r[key]) for r in selected) for key in ('success','fall','timeout','planning_failed')}
    summary[stratum].update(episodes=len(selected),sand_seconds=sum(r['sand_seconds'] for r in selected))
print(json.dumps(summary,indent=2))
(args.directory/'summary.json').write_text(json.dumps(summary,indent=2))
