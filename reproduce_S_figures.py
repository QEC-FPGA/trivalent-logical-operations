"""Reproduce the current S-gate figures from archived counts only."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
FIG10=FIG11=ROOT/'reproduced_figures'
FIG10.mkdir(exist_ok=True)
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
main=readcsv(ROOT/'data/counts/fig10_S_mainline_counts.csv')
r1=readcsv(ROOT/'data/counts/fig11b_W1_Wc_ratio_counts.csv')
r3=readcsv(ROOT/'data/counts/fig11c_W3_Wc_ratio_counts.csv')
for rows in [main,r1,r3]:
    for r in rows:
        r['d']=int(r['d'])
        for key in ['p','rate','ratio','lower95','upper95']:
            if key in r:r[key]=float(r[key])
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'path'})
cols=['#0072B2','#D55E00','#009E73','#8b4cb5','#c49800','#333333']
marks=['o','s','^','D','v','P'];styles=['-','--','-.',':',(0,(5,1,1,1)),(0,(3,1))]
def errorplot(ax,rows,x,y,color,marker,ls,label=None):
    rows=sorted(rows,key=lambda r:r[x])
    ax.errorbar([r[x] for r in rows],[r[y] for r in rows],
        yerr=np.array([[r[y]-r['lower95'],r['upper95']-r[y]] for r in rows]).T,
        color=color,marker=marker,linestyle=ls if len(rows)>1 else 'none',
        markersize=4,capsize=2.4,lw=1.1,elinewidth=.8,label=label)
def save(fig,folder,name):
    for ext in ['pdf','svg','png']:fig.savefig(folder/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(8.7,3.8),layout='constrained')
for ax,ch,title in zip(axs,['XY','ZZ'],[r'(a) $X\!\to\!Y$',r'(b) $Z\!\to\!Z$']):
    for j,d in enumerate([5,7,9,11,13]):
        rr=[r for r in main if r['d']==d and r['channel']==ch]
        errorplot(ax,rr,'p','rate',cols[j],marks[j],styles[j],f'$d={d}$')
    ax.set(xscale='log',yscale='log',xlabel=r'$p$',ylabel=r'$P_{\mathrm{L}}$',title=title,xlim=(.00043,.00465))
    ax.set_xticks([.0005,.001,.002,.004],labels=[r'$5\!\times\!10^{-4}$',r'$10^{-3}$',r'$2\!\times\!10^{-3}$',r'$4\!\times\!10^{-3}$'])
    ax.minorticks_off();ax.grid(alpha=.18);ax.legend(frameon=False,loc='lower right',ncol=2,fontsize=9.5,columnspacing=.8,handlelength=1.9)
save(fig,FIG10,'S_mainline_more_distances')

scan=readcsv(ROOT/'data/counts/fig11a_d13_Wscan_counts.csv')
for r in scan:
    for k in ['p','rate','lower95','upper95']:r[k]=float(r[k])
    r['W']=int(r['W'])

plt.rcParams.update({'font.family':['DejaVu Serif','SimSun'],'font.size':12,'axes.unicode_minus':False})
fig,axs=plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
for i,p in enumerate([.0005,.001,.002]):
    errorplot(axs[0],[r for r in scan if r['p']==p],'W','rate',cols[i],marks[i],styles[i],f'$p={p:g}$')
    errorplot(axs[1],[r for r in r1 if r['p']==p and r['d']<=13],'d','ratio',cols[i],marks[i],styles[i])
    errorplot(axs[2],[r for r in r3 if r['p']==p and r['d']<=13],'d','ratio',cols[i],marks[i],styles[i])
axs[0].set(xlabel=r'$W$',ylabel=r'$P_{\mathrm{L}}^{XY}$',title=r'(a) $d=13$',yscale='log',xlim=(.5,9.5),ylim=(3e-8,8e-3),xticks=[1,3,5,7,9])
axs[0].axvline(7,color='.55',ls=':',lw=.9);axs[0].text(7.12,1.1e-7,r'$W_c=7$',color='.4',fontsize=9)
axs[0].legend(frameon=False,loc='upper right',bbox_to_anchor=(.98,.83),fontsize=10)
for ax,title,y in [(axs[1],r'(b) $W1/W_c$',r'$R_{1/c}$'),(axs[2],r'(c) $W3/W_c$',r'$R_{3/c}$')]:
    ax.set(xlabel=r'$d$',ylabel=y,title=title,xlim=(4.5,13.5),ylim=(.78,1.56),xticks=[5,7,9,11,13])
    ax.axhline(1,color='.5',ls=':',lw=.9)
for ax in axs:ax.grid(alpha=.18,which='major')
save(fig,FIG11,'S_wait_comparison_abc')


