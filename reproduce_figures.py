"""Replot numerical figures from exact counts; no sampling or remote access."""
from pathlib import Path
import csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'reproduced_figures';DEST.mkdir(exist_ok=True)
def rows(name):
    with (ROOT/'data/counts'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def wilson(k,n):
    q=k/n;z=1.959963984540054;den=1+z*z/n
    c=(q+z*z/(2*n))/den;h=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)
def absolute(ax,data,xkey='p',label=''):
    data=sorted(data,key=lambda r:float(r[xkey]));x=[];y=[];lo=[];hi=[]
    for r in data:
        k,n=int(r['errors']),int(r['shots']);v=k/n;l,u=wilson(k,n)
        assert math.isclose(v,float(r['rate']),rel_tol=1e-12)
        x.append(float(r[xkey]));y.append(v);lo.append(v-l);hi.append(u-v)
    ax.errorbar(x,y,yerr=[lo,hi],marker='o',markersize=3,capsize=2,label=label)
def save(fig,stem):
    for suffix in ('pdf','svg'):fig.savefig(DEST/(stem+'.'+suffix))
    plt.close(fig)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
for name,channels,stem in [('fig09_H_counts.csv',['X-Z','Z-X'],'fig09_H'),('fig12_surgery_counts.csv',['ZZ','XX','XZ'],'fig12_surgery')]:
    data=rows(name);fig,axs=plt.subplots(1,len(channels),figsize=(4.7*len(channels),3.8),layout='constrained')
    for ax,ch in zip(axs,channels):
        for d in sorted({int(r['d']) for r in data}):absolute(ax,[r for r in data if r['channel']==ch and int(r['d'])==d],label=f'd={d}')
        ax.set(xscale='log',yscale='log',xlabel='Physical error probability p',ylabel='Logical error probability',title=ch)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,stem)

import runpy
runpy.run_path(str(ROOT/'reproduce_S_figures.py'),run_name='__main__')
print('H, S, waiting and surgery figures reproduced from archived counts.')
