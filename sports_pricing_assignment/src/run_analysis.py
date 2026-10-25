"""Reproducible preliminary simulation, using only the supplied synthetic CSV.

Run from the repository root: python src/run_analysis.py
No real aggregator records or external research data are downloaded.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json, time, platform, sys, hashlib, warnings, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats
import sklearn, scipy, lightgbm as lgb, optuna, torch, statsmodels
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
for x in ['tables', 'figures']: (OUT/x).mkdir(parents=True, exist_ok=True)
START = time.time()
CFG = dict(seeds=list(range(5)), trials=8, folds=4, gap_days=7,
           sequence_length=4, bootstrap=2000, mc_draws=1000,
           generator_seed=2026, tuning_seed=2026, bootstrap_seed=2026,
           owner_bounds=[.7, 1.3], cost_fraction=.3, cancellation_cost_fraction=.5,
           delta_r_min=0., min_group_n=500, min_residual_variance=.05)
(OUT/'config.json').write_text(json.dumps(CFG, indent=2))
def say(s): print(s, flush=True)
def save(d, name): d.to_csv(OUT/'tables'/f'{name}.csv', index=False)
def loss(y,p,kind='logloss'):
    p=np.clip(p,1e-8,1-1e-8)
    return (y-p)**2 if kind=='brier' else -(y*np.log(p)+(1-y)*np.log1p(-p))
def calibration(y,p):
    m=LogisticRegression(C=1e8,max_iter=500).fit(logit(np.clip(p,1e-6,1-1e-6))[:,None],y)
    return float(m.coef_[0,0]),float(m.intercept_[0])
def metrics(y,p,clim):
    slope,intercept=calibration(y,p)
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.mean(bins==b)*abs(y[bins==b].mean()-p[bins==b].mean()) for b in range(10) if np.any(bins==b))
    return dict(brier=loss(y,p,'brier').mean(),logloss=loss(y,p).mean(),auc=roc_auc_score(y,p),ece=ece,
                bss=1-loss(y,p,'brier').mean()/loss(y,clim,'brier').mean(),slope=slope,intercept=intercept)

say('Loading and auditing the supplied dataset')
raw=pd.read_csv(ROOT/'data/sports_bookings_synthetic.csv',parse_dates=['slot_start'])
dup=raw.duplicated(['venue_id','slot_start']); invalid=(raw.price<=0)|(raw.ref_price<=0)|raw[['price','ref_price']].isna().any(axis=1)
df=raw.loc[~dup&~invalid].sort_values(['slot_start','venue_id']).reset_index(drop=True)
assert not df.isna().any().any()
assert set(df.booked.unique()) <= {0,1}
assert (df.cancelled<=df.booked).all()
t=df.slot_start
df['hour']=t.dt.hour;df['dow']=t.dt.dayofweek;df['month']=t.dt.month
df['week']=t.dt.to_period('W').astype(str)
df['block']=df.venue_id.astype(str)+'_'+df.week
df['r']=df.price/df.ref_price;df['ln_r']=np.log(df.r)
df['peak']=df.hour.between(18,21).astype(int)
df['lead_bin']=pd.cut(df.lead_time_h,[-np.inf,3,24,168,np.inf],labels=['<3h','3-24h','1-7d','>7d']).astype(str)
df['group']=df.venue_type+'|'+df.lead_bin+'|'+df.peak.astype(str)
cut=df.slot_start.quantile(.8); train_idx=np.where(t<cut)[0]; test_idx=np.where(t>=cut)[0]
tr=df.iloc[train_idx].reset_index(drop=True);te=df.iloc[test_idx].reset_index(drop=True)
y=df.booked.to_numpy();yt=y[train_idx];ye=y[test_idx]
base=[]
for col,period in [('hour',24),('dow',7),('month',12)]:
    for fun,label in [(np.sin,'sin'),(np.cos,'cos')]:
        key=col+'_'+label;df[key]=fun(2*np.pi*df[col]/period);base.append(key)
base+=['temp','humidity','wind','precip','holiday','exam','lead_time_h']
# Category vocabulary is learned on training records. Unknown categories map to all zeros.
features=df[base].astype(float).copy()
for col in ['venue_type','venue_id']:
    for value in sorted(df.iloc[train_idx][col].unique()):features[f'{col}_{value}']=(df[col]==value).astype(float)
features['ln_r']=df.ln_r
# Every candidate receives the same four-slot covariate window, with no lagged outcomes.
seq=[]
for lag in range(CFG['sequence_length']-1,-1,-1):
    shifted=features.groupby(df.venue_id).shift(lag)
    seq.append(shifted.fillna(features).to_numpy(dtype=np.float32))
XS=np.stack(seq,axis=1);XF=XS.reshape(len(df),-1)
Xtr,Xte=XF[train_idx],XF[test_idx]
price_index=XF.shape[1]-1
Xref=XF.copy();Xref[:,price_index]=0
Xn=np.delete(XF,price_index,axis=1)
edges=pd.date_range(tr.slot_start.min(),tr.slot_start.max()+pd.Timedelta(seconds=1),periods=CFG['folds']+2)
folds=[];fr=[]
for i in range(1,CFG['folds']+1):
    a=np.where(tr.slot_start<edges[i]-pd.Timedelta(days=7))[0]
    b=np.where((tr.slot_start>=edges[i])&(tr.slot_start<edges[i+1]))[0]
    assert tr.slot_start.iloc[a].max()<tr.slot_start.iloc[b].min()-pd.Timedelta(days=7)
    folds.append((a,b));fr.append(dict(fold=i,train_n=len(a),validation_n=len(b),train_end=str(tr.slot_start.iloc[a].max()),validation_start=str(tr.slot_start.iloc[b].min()),validation_end=str(tr.slot_start.iloc[b].max())))
save(pd.DataFrame(fr),'validation_folds')
audit=dict(rows_before=len(raw),duplicate_slots=int(dup.sum()),invalid_prices=int(invalid.sum()),missing_cells=int(raw.isna().sum().sum()),rows_after=len(df),venues=df.venue_id.nunique(),start=str(t.min()),end=str(t.max()),train_n=len(tr),test_n=len(te),train_booked=int(yt.sum()),test_booked=int(ye.sum()),train_rate=float(yt.mean()),test_rate=float(ye.mean()),test_start=str(te.slot_start.min()),source_sha256=hashlib.sha256((ROOT/'data/sports_bookings_synthetic.csv').read_bytes()).hexdigest())
(OUT/'dataset_audit.json').write_text(json.dumps(audit,indent=2))
clim=tr.groupby(['venue_id','hour','dow']).booked.mean()
def climate(d):return pd.Series(list(zip(d.venue_id,d.hour,d.dow))).map(clim).fillna(yt.mean()).to_numpy()
pc=climate(te)

class LSTMModel:
    def __init__(self,p,seed):self.p=p;self.seed=seed
    def fit(self,x,y):
        torch.manual_seed(self.seed);np.random.seed(self.seed)
        self.scaler=StandardScaler().fit(x)
        a=self.scaler.transform(x).reshape(-1,CFG['sequence_length'],features.shape[1]).astype('float32')
        self.rnn=torch.nn.LSTM(features.shape[1],self.p['hidden'],batch_first=True)
        self.head=torch.nn.Linear(self.p['hidden'],1)
        params=list(self.rnn.parameters())+list(self.head.parameters())
        opt=torch.optim.Adam(params,lr=self.p['lr'],weight_decay=self.p['weight_decay'])
        a=torch.from_numpy(a);b=torch.tensor(np.asarray(y),dtype=torch.float32)
        rng=np.random.default_rng(self.seed)
        for ep in range(self.p['epochs']):
            order=rng.permutation(len(a))
            for start in range(0,len(a),2048):
                ix=order[start:start+2048];opt.zero_grad()
                z,_=self.rnn(a[ix]);pred=self.head(z[:,-1,:]).squeeze(-1)
                ll=torch.nn.functional.binary_cross_entropy_with_logits(pred,b[ix]);ll.backward()
                torch.nn.utils.clip_grad_norm_(params,5);opt.step()
        return self
    def predict_proba(self,x):
        a=self.scaler.transform(x).reshape(-1,CFG['sequence_length'],features.shape[1]).astype('float32');out=[]
        with torch.no_grad():
            for start in range(0,len(a),8192):
                z,_=self.rnn(torch.from_numpy(a[start:start+8192]));out.extend(torch.sigmoid(self.head(z[:,-1,:])).ravel().numpy())
        p=np.asarray(out);return np.column_stack([1-p,p])

def make_model(name,p,seed):
    if name=='Logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=p['C'],max_iter=500))
    if name=='LSTM':return LSTMModel(p,seed)
    mono=[0]*XF.shape[1];mono[-1]=-1
    return lgb.LGBMClassifier(**p,random_state=seed,n_jobs=2,verbosity=-1,monotone_constraints=mono,deterministic=True,force_col_wise=True)
best={};cv=[];tuning=[]
for name in ['Logistic','LightGBM','LSTM']:
    say('Tuning '+name+' on four chronological folds')
    def objective(trial):
        if name=='Logistic':p={'C':trial.suggest_float('C',1e-3,10,log=True)}
        elif name=='LightGBM':p=dict(n_estimators=trial.suggest_categorical('n_estimators',[120,240]),learning_rate=trial.suggest_float('learning_rate',.03,.1,log=True),num_leaves=trial.suggest_categorical('num_leaves',[7,15,31]),min_child_samples=trial.suggest_categorical('min_child_samples',[80,200]),reg_lambda=trial.suggest_float('reg_lambda',.1,10,log=True))
        else:p=dict(hidden=trial.suggest_categorical('hidden',[12,24]),lr=trial.suggest_float('lr',.001,.006,log=True),weight_decay=trial.suggest_float('weight_decay',1e-5,.01,log=True),epochs=trial.suggest_categorical('epochs',[5,8]))
        vals=[]
        for a,b in folds:
            m=make_model(name,p,0).fit(Xtr[a],yt[a]);vals.append(loss(yt[b],m.predict_proba(Xtr[b])[:,1]).mean())
        tuning.append(dict(model=name,trial=trial.number,mean_logloss=float(np.mean(vals)),parameters=json.dumps(p)))
        trial.set_user_attr('fold_scores',list(map(float,vals)))
        return float(np.mean(vals))
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=2026,n_startup_trials=4))
    study.optimize(objective,n_trials=CFG['trials'])
    best[name]=study.best_params
    cv.append(dict(model=name,mean_logloss=study.best_value,fold_sd=float(np.std(study.best_trial.user_attrs['fold_scores'],ddof=1)),fold_scores=json.dumps(study.best_trial.user_attrs['fold_scores'])))
    save(pd.DataFrame(tuning),'tuning');save(pd.DataFrame(cv),'cv_results')
    (OUT/'best_parameters.json').write_text(json.dumps(best,indent=2))
    say(name+' best CV log-loss '+str(study.best_value))
selected=min(cv,key=lambda x:x['mean_logloss'])['model'];say('Selected by validation: '+selected)
P={'Climatology':pc};SEEDS={};models={};rows=[]
for name in ['Logistic','LightGBM','LSTM']:
    say('Final five-seed fits: '+name)
    seeds=[0] if name=='Logistic' else CFG['seeds'];ms=[];pred=[]
    for seed in seeds:
        m=make_model(name,best[name],seed).fit(Xtr,yt);p=m.predict_proba(Xte)[:,1];pred.append(p);ms.append(m)
        rows.append(dict(model=name,seed=seed,**metrics(ye,p,pc)))
    models[name]=ms;SEEDS[name]=np.array(pred);P[name]=np.mean(pred,axis=0)
rows.append(dict(model='Climatology',seed=0,**metrics(ye,pc,pc)))
save(pd.DataFrame(rows),'per_seed_metrics')

# SARIMAX is estimated at the planned venue-hour occupancy level. Each series has one slot per hour.
# Forecasts use covariates only throughout the test horizon; no test outcomes are assimilated.
say('Fitting SARIMAX baselines with training AIC selection')
sar=np.zeros(len(te));srows=[]
excols=['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','temp','humidity','wind','precip','holiday','exam','lead_time_h','ln_r']
for venue in sorted(tr.venue_id.unique()):
    a=np.where(tr.venue_id==venue)[0];b=np.where(te.venue_id==venue)[0]
    scaler=StandardScaler().fit(df.iloc[train_idx[a]][excols])
    xa=scaler.transform(df.iloc[train_idx[a]][excols]);xb=scaler.transform(df.iloc[test_idx[b]][excols])
    candidates=[]
    for order in [(0,0,0),(1,0,0)]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit=SARIMAX(yt[a],exog=xa,order=order,seasonal_order=(1,0,0,15),trend='c',enforce_stationarity=True).fit(disp=False,maxiter=60)
        candidates.append(fit)
    fit=min(candidates,key=lambda m:m.aic)
    sar[b]=np.clip(np.asarray(fit.forecast(len(b),exog=xb)),1e-6,1-1e-6)
    srows.append(dict(venue=int(venue),order=str(fit.model.order),seasonal_order=str(fit.model.seasonal_order),aic=fit.aic,converged=bool(fit.mle_retvals['converged'])))
    say('SARIMAX venue '+str(venue)+' complete')
P['SARIMAX']=sar;save(pd.DataFrame(srows),'sarimax_fits')

say('Forecast uncertainty and paired tests')
bc=pd.factorize(te.block)[0];nblocks=bc.max()+1;rng=np.random.default_rng(2026)
counts=np.bincount(bc);draw_index=rng.integers(0,nblocks,size=(CFG['bootstrap'],nblocks))
def boot_means(v):
    sums=np.bincount(bc,weights=v);return sums[draw_index].sum(1)/counts[draw_index].sum(1)
summary=[]
for name,p in P.items():
    mm=metrics(ye,p,pc);r=dict(model=name,**mm)
    for key in ['brier','logloss']:
        lo,hi=np.quantile(boot_means(loss(ye,p,key)),[.025,.975]);r[key+'_lo']=lo;r[key+'_hi']=hi
    daily=pd.DataFrame({'venue':te.venue_id,'day':te.slot_start.dt.date,'y':ye,'p':p}).groupby(['venue','day'])[['y','p']].sum()
    r['mae_bookings_day']=abs(daily.y-daily.p).mean();r['wape_percent']=100*abs(daily.y-daily.p).sum()/daily.y.sum()
    summary.append(r)
save(pd.DataFrame(summary),'forecast_metrics')
tests=[]
for name,p in P.items():
    if name==selected:continue
    for kind in ['brier','logloss']:
        delta=loss(ye,P[selected],kind)-loss(ye,p,kind);samples=boot_means(delta);lo,hi=np.quantile(samples,[.025,.975])
        # Null-centred paired cluster bootstrap, finite-simulation correction.
        pv=(1+np.sum(np.abs(samples-delta.mean())>=abs(delta.mean())))/(len(samples)+1)
        tests.append(dict(baseline=name,metric=kind,difference=delta.mean(),ci_lo=lo,ci_hi=hi,p=pv))
pv=np.array([r['p'] for r in tests]);order=np.argsort(pv);adj=np.empty(len(pv));adj[order]=np.minimum(1,np.maximum.accumulate((len(pv)-np.arange(len(pv)))*pv[order]))
for r,p in zip(tests,adj):r['p_holm']=p
save(pd.DataFrame(tests),'paired_tests')
idxblocks=[np.where(bc==i)[0] for i in range(nblocks)];sl=[]
for ks in draw_index[:300]:
    ix=np.concatenate([idxblocks[k] for k in ks]);sl.append(calibration(ye[ix],P[selected][ix])[0])
slope_ci=list(map(float,np.quantile(sl,[.025,.975])))

say('Past-only cross-fitting for elasticity')
# First block is warm-up. Four expanding estimation folds keep the seven-day gap.
vy=np.full(len(tr),np.nan);vt=np.full(len(tr),np.nan)
for a,b in folds:
    kw=dict(n_estimators=160,num_leaves=15,min_child_samples=100,learning_rate=.05,n_jobs=2,verbosity=-1,random_state=2026,deterministic=True,force_col_wise=True)
    gm=lgb.LGBMRegressor(**kw).fit(Xn[train_idx[a]],yt[a]);pm=lgb.LGBMRegressor(**kw).fit(Xn[train_idx[a]],tr.ln_r.iloc[a])
    vy[b]=yt[b]-gm.predict(Xn[train_idx[b]]);vt[b]=tr.ln_r.iloc[b]-pm.predict(Xn[train_idx[b]])
er=[]
truth=pd.read_csv(ROOT/'data/ground_truth.csv');trtruth=tr.merge(truth,on='slot_id',validate='one_to_one')
truth_prob=expit(trtruth.logit0_true+trtruth.beta_true*trtruth.ln_r)
true_theta=trtruth.beta_true.to_numpy()*truth_prob.to_numpy()*(1-truth_prob.to_numpy())
for g,ix0 in tr.groupby('group').indices.items():
    ix=ix0[np.isfinite(vt[ix0])];v=vt[ix];z=vy[ix];den=np.sum(v*v);th=np.sum(v*z)/den
    psi=v*(z-th*v);bs=pd.Series(psi).groupby(tr.block.iloc[ix].to_numpy()).sum().to_numpy()
    se=np.sqrt(np.sum(bs**2))/den;dbar=yt[ix].mean();share=np.var(v)/np.var(tr.ln_r.iloc[ix])
    naive=np.cov(tr.ln_r.iloc[ix],yt[ix],ddof=0)[0,1]/np.var(tr.ln_r.iloc[ix])
    er.append(dict(group=g,n=len(ix),theta=th,se=se,lo=th-1.96*se,hi=th+1.96*se,eps=th/dbar,eps_se=se/dbar,unexplained_share=share,naive_theta=naive,true_theta=np.mean(true_theta[ix]),used=bool(len(ix)>=500 and share>=.05)))
el=pd.DataFrame(er).set_index('group');save(el.reset_index(),'elasticity')
eps=tr.group.map(el.eps).to_numpy();epste=te.group.map(el.eps).to_numpy();es=te.group.map(el.eps_se).to_numpy()
used=te.group.map(el.used).to_numpy();epste[~used]=0;es[~used]=0

say('Reference-price demand, cancellations and policy evaluation')
D0tr=np.mean([m.predict_proba(Xref[train_idx])[:,1] for m in models[selected]],axis=0)
D0te=np.mean([m.predict_proba(Xref[test_idx])[:,1] for m in models[selected]],axis=0)
# Training-only revenue normalization; clipping is included while solving for the factor.
from scipy.optimize import brentq
base_tr=D0tr*tr.r.to_numpy()**eps
target=np.sum(tr.price.to_numpy()*yt)
factor=brentq(lambda s:np.sum(tr.price.to_numpy()*np.minimum(1,s*base_tr))-target,.01,10)
D0=D0te*factor
# Simulator cancellation rule has a price threshold. Flexible classifier learns it from booked training slots.
bk=yt==1
qmodel=lgb.LGBMClassifier(n_estimators=120,num_leaves=7,min_child_samples=200,learning_rate=.04,n_jobs=2,verbosity=-1,random_state=2026).fit(Xtr[bk],tr.cancelled.to_numpy()[bk])
qold=qmodel.predict_proba(Xte)[:,1]
r0=te.r.to_numpy();ref=te.ref_price.to_numpy();rnew=r0.copy();qnew=qold.copy()
support={g:np.quantile(tr.r.iloc[ix],[.025,.975]) for g,ix in tr.groupby('group').indices.items()}
for g,ix in te.groupby('group').indices.items():
    if not el.loc[g,'used']:continue
    lo,hi=support[g];grid=np.arange(max(70,int(np.ceil(lo*100))),min(130,int(np.floor(hi*100)))+1)/100
    # Historical price is retained as an explicit no-change option even outside trimmed support.
    bestprofit=(ref[ix]*(r0[ix]-.3-.5*qold[ix]))*np.minimum(1,D0[ix]*r0[ix]**epste[ix])
    for r in grid:
        xx=Xte[ix].copy();xx[:,-1]=np.log(r);q=qmodel.predict_proba(xx)[:,1]
        dem=np.minimum(1,D0[ix]*r**epste[ix]);profit=ref[ix]*(r-.3-.5*q)*dem
        better=profit>bestprofit;ii=ix[better];rnew[ii]=r;qnew[ii]=q[better];bestprofit[better]=profit[better]
p0=ref*r0;pn=ref*rnew
off=(te.dow.to_numpy()<5)&(te.hour.to_numpy()>=10)&(te.hour.to_numpy()<16)
def demand(e):return np.minimum(1,D0*r0**e),np.minimum(1,D0*rnew**e)
def totals(e):
    a,b=demand(e)
    # Ten sufficient statistics per venue-week block.
    v=[p0*a,pn*b,a,b,off*a,off*b,a*qold,b*qnew,np.ones(len(a)),off.astype(float)]
    return np.array([np.bincount(bc,weights=x,minlength=nblocks) for x in v])
def effects(v):
    return np.array([100*(v[1]/v[0]-1),100*(v[3]-v[2])/v[8],100*(v[5]-v[4])/v[9],100*(v[7]/v[3]-v[6]/v[2])])
point=effects(totals(epste).sum(axis=1));rng=np.random.default_rng(123);draws=[]
groups=list(el.index);gm=np.array([groups.index(g) for g in te.group]);eh=el.eps.to_numpy();ese=el.eps_se.to_numpy();eu=el.used.to_numpy()
for _ in range(CFG['mc_draws']):
    e=np.where(eu,eh+rng.normal(size=len(eh))*ese,0)[gm]
    ks=rng.integers(0,nblocks,size=nblocks);draws.append(effects(totals(e)[:,ks].sum(axis=1)))
interval=np.quantile(draws,[.025,.975],axis=0)
policy=pd.DataFrame({'metric':['revenue_percent','utilization_pp','offpeak_pp','cancellation_pp'],'estimate':point,'lo':interval[0],'hi':interval[1]});save(policy,'policy_effects')
te_truth=te.merge(truth,on='slot_id',validate='one_to_one')
trueold=expit(te_truth.logit0_true.to_numpy()+te_truth.beta_true.to_numpy()*np.log(r0));truenew=expit(te_truth.logit0_true.to_numpy()+te_truth.beta_true.to_numpy()*np.log(rnew))
qtold=.06+.10*(r0>1.1)+.03*(te.lead_time_h.to_numpy()>48)
qtnew=.06+.10*(rnew>1.1)+.03*(te.lead_time_h.to_numpy()>48)
truev=np.array([np.sum(p0*trueold),np.sum(pn*truenew),trueold.sum(),truenew.sum(),np.sum(off*trueold),np.sum(off*truenew),np.sum(trueold*qtold),np.sum(truenew*qtnew),len(te),off.sum()])
truefx=effects(truev);save(pd.DataFrame({'metric':policy.metric,'truth':truefx}),'policy_truth')
ks=np.linspace(.5,1.5,11);sens=np.array([effects(totals(epste*k).sum(axis=1))[0] for k in ks]);save(pd.DataFrame({'scale':ks,'revenue_percent':sens}),'sensitivity')
wk=tr[tr.booked==1].groupby('week').cancelled.mean()*100;margin=float(.5*wk.std())
a,b=demand(epste)

# Additional temporal/group errors and weather ablation use the same fixed test records.
gr=[]
for col,values in [('venue_type',te.venue_type),('season',te.month.map({1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn',12:'winter'}))]:
    for g in sorted(values.unique()):
        ix=np.where(values==g)[0];gr.append(dict(grouping=col,group=g,n=len(ix),booked=int(ye[ix].sum()),brier=loss(ye[ix],P[selected][ix],'brier').mean(),logloss=loss(ye[ix],P[selected][ix]).mean(),mean_pred=P[selected][ix].mean(),observed=ye[ix].mean()))
save(pd.DataFrame(gr),'group_errors')
weather=[features.columns.get_loc(c) for c in ['temp','humidity','wind','precip']]
remove=[j*features.shape[1]+c for j in range(4) for c in weather];keep=np.setdiff1d(np.arange(XF.shape[1]),remove)
# A separate LightGBM ablation avoids changing the selected model after test inspection.
say('Weather ablation')
abpred=[]
for seed in CFG['seeds']:
    mono=[0]*len(keep);mono[-1]=-1
    m=lgb.LGBMClassifier(**best['LightGBM'],random_state=seed,n_jobs=2,verbosity=-1,monotone_constraints=mono,deterministic=True,force_col_wise=True).fit(Xtr[:,keep],yt)
    abpred.append(m.predict_proba(Xte[:,keep])[:,1])
pab=np.mean(abpred,axis=0);ab=[]
for kind in ['brier','logloss']:
    d=loss(ye,pab,kind)-loss(ye,P['LightGBM'],kind);lo,hi=np.quantile(boot_means(d),[.025,.975]);ab.append(dict(metric=kind,without_weather=float(loss(ye,pab,kind).mean()),with_weather=float(loss(ye,P['LightGBM'],kind).mean()),difference=float(d.mean()),lo=lo,hi=hi))
save(pd.DataFrame(ab),'weather_ablation')

predictions=te[['slot_id','slot_start','venue_id','venue_type','booked','cancelled','price','ref_price']].copy()
for name,p in P.items():predictions['p_'+name]=p
predictions['recommended_price']=pn;predictions['demand_historical']=a;predictions['demand_recommended']=b
predictions.to_csv(OUT/'test_predictions.csv.gz',index=False,compression='gzip')

import runpy
runpy.run_path(str(ROOT/'src/plot_saved_results.py'), run_name='__main__')

# Success checks and execution metadata; no prospective power calculation is claimed.
margin_h1=all(r['difference']<0 and r['p_holm']<.05 for r in tests) and metrics(ye,P[selected],pc)['bss']>0 and slope_ci[0]<=1<=slope_ci[1]
summary_meta=dict(selected=selected,slope_ci=slope_ci,n_test_blocks=int(nblocks),calibration_factor=factor,
    historical_revenue_diagnostic=float(np.sum(p0*a)/np.sum(p0*ye)),historical_demand_rate=float(a.mean()),
    recommended_rate=float(b.mean()),mean_price_ratio=float(np.mean(rnew/r0)),changed_share=float(np.mean(rnew!=r0)),
    cancellation_margin_pp=margin,truth_effects=list(map(float,truefx)),
    positive_theta=int((el.theta>0).sum()),elasticity_coverage=float(((el.true_theta>=el.lo)&(el.true_theta<=el.hi)).mean()),
    theta_mae=float(abs(el.theta-el.true_theta).mean()),naive_mae=float(abs(el.naive_theta-el.true_theta).mean()),
    residual_share_min=float(el.unexplained_share.min()),residual_share_max=float(el.unexplained_share.max()),
    n_used_segments=int(el.used.sum()),crossfit_records=int(np.isfinite(vt).sum()),
    h1=bool(margin_h1),h2=bool(interval[0,0]>0),h3=bool(interval[0,2]>0 and interval[1,3]<margin),
    runtime_seconds=time.time()-START,python=sys.version.split()[0],platform=platform.platform(),cpu_count=os.cpu_count(),
    cpu_model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
    versions={m.__name__:m.__version__ for m in [np,pd,sklearn,scipy,lgb,optuna,torch,statsmodels]},
    sarimax_converged=sum(r['converged'] for r in srows))
(OUT/'summary.json').write_text(json.dumps(summary_meta,indent=2))
say(json.dumps(summary_meta,indent=2));say('Analysis finished')
