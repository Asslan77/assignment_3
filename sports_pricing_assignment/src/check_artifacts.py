"""Check saved predictions against reported metrics and policy constraints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/sports_bookings_synthetic.csv',parse_dates=['slot_start'])
p=pd.read_csv(ROOT/'results/test_predictions.csv.gz',parse_dates=['slot_start'])
m=pd.read_csv(ROOT/'results/tables/forecast_metrics.csv')
a=json.loads((ROOT/'results/dataset_audit.json').read_text())
assert len(df)==a['rows_after']==153300
assert not df.duplicated(['venue_id','slot_start']).any()
assert len(p)==a['test_n']==30660
assert set(p.slot_id)==set(df[df.slot_start>=df.slot_start.quantile(.8)].slot_id)
for _,r in m.iterrows():
    pr=p['p_'+r.model].to_numpy();y=p.booked.to_numpy()
    assert np.isfinite(pr).all() and ((pr>=0)&(pr<=1)).all()
    assert np.isclose(np.mean((y-pr)**2),r.brier,atol=1e-12)
    pr=np.clip(pr,1e-8,1-1e-8)
    assert np.isclose(-np.mean(y*np.log(pr)+(1-y)*np.log1p(-pr)),r.logloss,atol=1e-12)
changed=~np.isclose(p.price,p.recommended_price)
rr=p.recommended_price/p.ref_price
assert rr[changed].between(.7-1e-9,1.3+1e-9).all()
assert np.isclose(rr[changed]*100,np.round(rr[changed]*100)).all()
policy=pd.read_csv(ROOT/'results/tables/policy_effects.csv')
gain=100*((p.recommended_price*p.demand_recommended).sum()/(p.price*p.demand_historical).sum()-1)
assert np.isclose(gain,policy.iloc[0].estimate,atol=1e-10)
fold=pd.read_csv(ROOT/'results/tables/validation_folds.csv')
assert ((pd.to_datetime(fold.validation_start)-pd.to_datetime(fold.train_end))>pd.Timedelta(days=7)).all()
print('PASS: dataset, test IDs, probability metrics, chronological gaps, price grid and revenue arithmetic')
