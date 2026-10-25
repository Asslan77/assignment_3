"""Regenerate the article's vector figures from the saved analysis outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
p=pd.read_csv(OUT/'test_predictions.csv.gz')
selected=pd.read_csv(OUT/'tables/cv_results.csv').sort_values('mean_logloss').iloc[0].model
el=pd.read_csv(OUT/'tables/elasticity.csv')
sens=pd.read_csv(OUT/'tables/sensitivity.csv')
truth=pd.read_csv(OUT/'tables/policy_truth.csv').set_index('metric').loc['revenue_percent','truth']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.4,2.65))
ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Perfect calibration')
y=p.booked.to_numpy()
for name in [selected,'Logistic','SARIMAX']:
    v=p['p_'+name].to_numpy();bn=np.minimum((v*10).astype(int),9)
    xx=[v[bn==j].mean() for j in range(10) if np.sum(bn==j)>=20]
    yy=[y[bn==j].mean() for j in range(10) if np.sum(bn==j)>=20]
    ax.plot(xx,yy,'o-',ms=3,label=name)
ax.set(xlabel='Mean predicted booking probability',ylabel='Observed booking frequency',xlim=(0,1),ylim=(0,1))
ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'figures/reliability.pdf');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7,2.6))
axs[0].errorbar(np.arange(len(el)),el.theta,yerr=1.96*el.se,fmt='o',ms=2,label='DML estimate and 95% interval')
axs[0].plot(np.arange(len(el)),el.true_theta,'x',ms=3,label='Simulator derivative');axs[0].axhline(0,color='gray',lw=.7)
axs[0].set(xlabel='Segment index (see elasticity.csv)',ylabel='Probability derivative\nper unit ln(price ratio)')
axs[0].legend(fontsize=6)
axs[1].plot(sens.scale,sens.revenue_percent,'o-',ms=3,label='Estimated response')
axs[1].axhline(truth,ls='--',color='darkred',label='Simulator policy effect');axs[1].axhline(0,color='gray',lw=.7)
axs[1].set(xlabel='Elasticity multiplier (dimensionless)',ylabel='Revenue change (%)');axs[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'figures/elasticity_sensitivity.pdf');plt.close(fig)

gg=pd.read_csv(OUT/'tables/group_errors.csv');gg=gg[gg.grouping=='venue_type']
fig,ax=plt.subplots(figsize=(3.4,2.6));x=np.arange(len(gg))
ax.bar(x-.17,gg.observed*100,.34,label='Observed');ax.bar(x+.17,gg.mean_pred*100,.34,label='Predicted')
ax.set_xticks(x,gg.group);ax.set_ylabel('Booking rate (%)')
ax.legend(fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
fig.tight_layout();fig.savefig(OUT/'figures/group_errors.pdf');plt.close(fig)
