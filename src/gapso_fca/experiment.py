"""CLI: run paper-scale protocol or a small deterministic smoke reproduction."""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import ASSUMPTIONS, Params, Structure, fitness, metrics, reference_field, simulate
from .optimize import SearchConfig, optimize

MODELS = ('CA', 'FCA', 'GA-FCA', 'PSO-FCA', 'GA-PSO-FCA')
CONDITIONS = {
    'isotropic': (0.0016, 0.0016, 0.0016),
    'anisotropic': (0.0022, 0.0015, 0.0009),
}


def settings(mode: str):
    if mode == 'paper':
        return dict(n_cal=30, n_val=50, steps=(10,20,30,40,50),
                    calibration=(10,20,30), validation=(40,50),
                    seeds=20, cfg=SearchConfig())
    if mode == 'verification':
        return dict(n_cal=30, n_val=50, steps=(10,20,30,40,50),
                    calibration=(10,20,30), validation=(40,50), seeds=1,
                    cfg=SearchConfig(ga_population=4,ga_generations=2,
                                     pso_particles=4,pso_iterations=2))
    if mode == 'smoke':
        return dict(n_cal=10, n_val=16, steps=(2,4,6,8,10),
                    calibration=(2,4,6), validation=(8,10), seeds=1,
                    cfg=SearchConfig(ga_population=4,ga_generations=2,
                                     pso_particles=4,pso_iterations=2))
    raise ValueError(mode)


def save_figures(path: Path, condition: str, fields, references, summaries, histories, selected_step):
    names = ['Reference']+list(MODELS)
    fig, axes = plt.subplots(2,len(names),figsize=(3*len(names),6),constrained_layout=True)
    for j,name in enumerate(names):
        field = references[selected_step] if name=='Reference' else fields[name]
        mid=field.shape[2]//2
        for row, plane in enumerate((field[:,:,mid],field[:,mid,:])):
            axes[row,j].imshow(plane.T,origin='lower',cmap='viridis',vmin=0,vmax=max(references[selected_step].max(),1e-15))
            axes[row,j].set_title(f'{name} / {"XY" if row==0 else "XZ"}',fontsize=9)
            axes[row,j].set_xticks([]); axes[row,j].set_yticks([])
    fig.suptitle(condition+' : reference vs 5 methods')
    fig.savefig(path/f'{condition}_slices.png',dpi=140)
    plt.close(fig)
    fig, axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    labels=list(MODELS)
    for ax, metric, ylabel in zip(axs, ('rmse','r2'), ('RMSE','R²')):
        vals=[summaries[k][metric] for k in labels]
        ax.bar(np.arange(5),vals)
        ax.set_xticks(np.arange(5),labels,rotation=25,ha='right')
        ax.set_ylabel(ylabel);ax.set_title(condition)
        if metric=='rmse': ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    fig.savefig(path/f'{condition}_metrics.png',dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for model,history in histories.items():
        if history:
            ax.plot(np.arange(1,len(history)+1),history,label=model)
    ax.set_xlabel('Generation / iteration');ax.set_ylabel('Calibration fitness (best so far)')
    ax.set_title(f'{condition}: convergence, representative seed');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(path/f'{condition}_convergence.png',dpi=150);plt.close(fig)
    fig = plt.figure(figsize=(10,9))
    for k,(name,field) in enumerate((('Reference',references[selected_step]),('GA-PSO-FCA',fields['GA-PSO-FCA']))):
        ax = fig.add_subplot(1,2,k+1,projection='3d')
        normalized = field/max(float(field.max()),1e-30)
        xyz=np.array(np.where(normalized>=.03))
        dots=normalized[normalized>=.03]
        ax.scatter(xyz[0],xyz[1],xyz[2],c=dots,cmap='viridis',s=4,alpha=.55,vmin=0,vmax=1)
        ax.set_title(condition+' / '+name)
        for label_axis in (ax.xaxis,ax.yaxis,ax.zaxis): label_axis.set_tick_params(labelsize=7)
    fig.savefig(path/f'{condition}_3d.png',dpi=140);plt.close(fig)


def save_grid_sensitivity(out: Path, dt: float, spec):
    # Fig. 2-type check uses the ordinary CA, the paper's strongest baseline.
    # Fixed validation times across all three grid sizes.
    rows=[]
    for condition,D in CONDITIONS.items():
        for n in (30,40,50):
            if max(spec['validation']) < 40:  # use the demo's configured times
                eval_times=spec['validation']
            else:
                eval_times=(40,50)
            sim=simulate(n,max(eval_times),dt,'CA',true_diffusion=D,snapshots=eval_times)
            rmse=np.mean([metrics(sim[t],reference_field(n,t,dt,D))['rmse'] for t in eval_times])
            rows.append(dict(condition=condition,grid=n,rmse=float(rmse)))
    with (out/'grid_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['condition','grid','rmse']);writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(7,4))
    for condition in CONDITIONS:
        rr=[r for r in rows if r['condition']==condition]
        ax.plot([r['grid'] for r in rr],[r['rmse'] for r in rr],'o-',label=condition)
    ax.set_xlabel('Grid size N (N³ cells)');ax.set_ylabel('CA validation RMSE')
    ax.set_title('Grid sensitivity (computed, not reported values)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(out/'grid_sensitivity.png',dpi=150);plt.close(fig)

def run(mode: str='smoke', out='results/runs/smoke', seeds: int | None = None, no_figures: bool = False, condition_only: str | None = None):
    spec=settings(mode)
    if seeds is not None:
        if seeds < 1: raise ValueError('seeds must be positive')
        spec['seeds'] = seeds
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=spec['cfg']; dt=.02
    run_rows, model_records, global_summary={}, [], {}
    config = dict(mode=mode, dt=dt, n_cal=spec['n_cal'],n_val=spec['n_val'],
                  calibration=spec['calibration'],validation=spec['validation'],
                  independent_seeds=spec['seeds'],ga=vars(cfg),
                  assumptions=ASSUMPTIONS, note='Measured outputs; not copied from paper tables')
    (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    for condition,D in CONDITIONS.items():
        if condition_only is not None and condition != condition_only: continue
        print(f'Condition: {condition}, D={D}',flush=True)
        reference_cal={k:reference_field(spec['n_cal'],k,dt,D) for k in spec['calibration']}
        reference_val={k:reference_field(spec['n_val'],k,dt,D) for k in spec['validation']}
        records=[]; fields={}; histories={}; selected=[]
        for seed in range(spec['seeds']):
            for model in MODELS:
                t0=time.perf_counter()
                structure,params=Structure(),Params()
                hist=[]; n_unique=0
                if model not in ('CA','FCA'):
                    def obj(s,p): return fitness(s,p,spec['n_cal'],dt,spec['calibration'],reference_cal)
                    structure,params,_,hist,n_unique=optimize(model,obj,cfg,seed)
                opt_time=time.perf_counter()-t0
                t1=time.perf_counter()
                saved=simulate(spec['n_val'],max(spec['validation']),dt,model,structure,params,
                               D,spec['validation'])
                sim_time=time.perf_counter()-t1
                vals=[metrics(saved[t],reference_val[t]) for t in spec['validation']]
                avg={key:float(np.mean([x[key] for x in vals])) for key in vals[0]}
                row=dict(condition=condition,method=model,seed=seed,**avg,
                         optimization_seconds=opt_time,simulation_seconds=sim_time,
                         neighborhood=structure.neighborhood,rule=structure.rule+1,
                         distance_power=structure.distance_power,unique_structures=n_unique,
                         **{key:float(z) for key,z in zip(('Dx','Dy','Dz','lambda_s','m_f','w_f'),params.as_array())})
                records.append(row);model_records.append(row)
                print(f'  {model:12s} seed={seed:02d} RMSE={avg["rmse"]:.6g} R2={avg["r2"]:.5f} opt={opt_time:.2f}s',flush=True)
                if seed==0:
                    fields[model]=saved[max(spec['validation'])]
                    histories[model]=hist
                if model=='GA-PSO-FCA':selected.append(str(structure.key()))
        summarized={}
        for model in MODELS:
            matching=[x for x in records if x['method']==model]
            summarized[model]={metric:float(np.mean([x[metric] for x in matching])) for metric in
                               ('rmse','mae','r2','mass_error','optimization_seconds','simulation_seconds')}
            summarized[model].update({metric+'_std':float(np.std([x[metric] for x in matching],ddof=1)) if len(matching)>1 else 0.
                                      for metric in ('rmse','mae','r2')})
        global_summary[condition]=summarized
        (out/f'{condition}_selections.json').write_text(json.dumps(dict(Counter(selected)),indent=2))
        if not no_figures:
            save_figures(out,condition,fields,reference_val,summarized,histories,max(spec['validation']))
    with (out/'individual_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=model_records[0].keys());writer.writeheader();writer.writerows(model_records)
    (out/'summary.json').write_text(json.dumps(global_summary,indent=2),encoding='utf-8')
    if not no_figures:
        save_grid_sensitivity(out, dt, spec)
    return global_summary


def main():
    parser=argparse.ArgumentParser(description='Reproduce GA–PSO 3D FCA diffusion experiments')
    parser.add_argument('--mode',choices=('smoke','verification','paper'),default='smoke')
    parser.add_argument('--out',type=str,default=None)
    parser.add_argument('--seeds',type=int,default=None,help='Override the number of independent runs')
    parser.add_argument('--no-figures',action='store_true',help='Skip expensive plotting; still save CSV and JSON')
    parser.add_argument('--condition',choices=tuple(CONDITIONS),default=None,help='Run one diffusion condition')
    args=parser.parse_args()
    directory=args.out or f'results/runs/{args.mode}'
    run(args.mode,directory,args.seeds,args.no_figures,args.condition)

if __name__=='__main__':main()
