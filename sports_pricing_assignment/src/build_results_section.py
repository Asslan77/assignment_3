"""Build the Results prose and tables from the saved, checked run outputs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';P=ROOT/'paper'
s=json.loads((OUT/'summary.json').read_text());a=json.loads((OUT/'dataset_audit.json').read_text())
f=pd.read_csv(OUT/'tables/forecast_metrics.csv').set_index('model')
cv=pd.read_csv(OUT/'tables/cv_results.csv').set_index('model')
tests=pd.read_csv(OUT/'tables/paired_tests.csv')
pol=pd.read_csv(OUT/'tables/policy_effects.csv').set_index('metric')
tru=pd.read_csv(OUT/'tables/policy_truth.csv').set_index('metric')
ab=pd.read_csv(OUT/'tables/weather_ablation.csv').set_index('metric')
el=pd.read_csv(OUT/'tables/elasticity.csv')
folds=pd.read_csv(OUT/'tables/validation_folds.csv')
seeds=pd.read_csv(OUT/'tables/per_seed_metrics.csv')
sens=pd.read_csv(OUT/'tables/sensitivity.csv')
g=pd.read_csv(OUT/'tables/group_errors.csv').set_index('group')
def n(v,k=4):return f'{v:.{k}f}'
def ci(row,k=2):return f"{row['lo']:.{k}f} to {row['hi']:.{k}f}"
def table(caption,label,cols,header,rows,wide=False,size=r'\footnotesize'):
    env='table*' if wide else 'table'
    return '\n'.join([r'\begin{'+env+'}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',size,r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{'+env+'}'])
parts=[r'''\section{Results}
\label{sec:results}
\subsection{Study status and research questions}
This preliminary study evaluated the forecasting and pricing pipeline on the supplied synthetic dataset. RQ1 concerns booking-probability accuracy, RQ2 concerns the counterfactual revenue change from constrained prices, and RQ3 concerns off-peak occupancy and cancellations. Each question was assessed against the corresponding H1--H3 criterion defined in Methods. All models and policies are evaluated on the same final test period. The findings describe this simulator and the models fitted to it; no real aggregator records were used.

The analysis covers the full supplied synthetic dataset. These are measured simulation results, calculated with the accompanying code and retained in the saved outputs. These preliminary findings come from completed model runs on synthetic records; they are not projected outcomes or measurements from real bookings. The simulator's hidden parameters were excluded from training, model selection and price optimization. Only after fitting were these parameters used to check the final predictions and pricing policy against the known booking mechanism.

\subsection{Data actually used and deviations}
Table~\ref{tab:data} records the final analysis data. The 153,300 rows cover 730 days and 14 venues, with 15 offered slots per venue-day. The first 584 days form the development period, and the last 146 days form the test period. There are 31,888 booked and 90,752 unbooked development slots, compared with 9,988 booked and 20,672 unbooked test slots. Booking prevalence is 26.00\% and 32.58\%, respectively. The development period contained 3,187 cancellations among booked slots, and the test period contained 958. These correspond to 9.99\% and 9.59\% of bookings. Cancelled bookings remain positive examples for the booking target; the cancellation classifier models their later cancellation separately. No cancelled rows were discarded from the forecasting comparison. All 15 supplied columns are complete. The audit found no duplicate venue--time keys, nonpositive prices or missing prices, so all records were retained.

The generator reproduced the supplied CSV byte for byte with seed 2026. The same generation run recovered the hidden booking intercepts and price coefficients used in the simulator checks. Training windows expand across the development folds and therefore overlap. Adding their training and validation counts would count some observations more than once. Each validation fold is separated from its training observations by seven days. The held-out test observations occur from 8 April to 31 August 2026 and contain spring and summer dates only.
''']
rows=[['Source','Supplied synthetic booking simulator'],['Period','2024-09-01 to 2026-08-31'],['Venues / facility types','14 / 4'],['Rows before / after cleaning','153,300 / 153,300'],['Duplicates / invalid prices / missing cells','0 / 0 / 0'],['Development / test rows','122,640 / 30,660'],['Booked / unbooked development rows','31,888 / 90,752'],['Booked / unbooked test rows','9,988 / 20,672'],['Booking prevalence development / test',r'26.00\% / 32.58\%']]
for _,r in folds.iterrows():rows.append([f"Fold {int(r.fold)} train / validation",f'{r.train_n:,} / {r.validation_n:,}'])
parts.append(table('Final synthetic dataset and chronological validation sizes. Development includes all pre-test observations; fold windows overlap. Zero rows were removed.','tab:data',r'p{.46\columnwidth}p{.46\columnwidth}','Item & Value',rows,size=r'\footnotesize'))
parts.append(r'''Several parts of the original Methods were reduced or changed for this pilot. Real aggregator data and external weather observations are replaced by the supplied simulation. Hyperparameter search uses eight trials per learned candidate instead of 100. Training uses a CPU, and the LSTM has a fixed five- or eight-epoch search budget rather than early stopping. The neural comparison includes an LSTM but no Transformer. SARIMAX uses a restricted AIC order search and a fixed multi-step test forecast. The reported comparisons therefore cover the pilot search budget; the larger tuning study remains unfinished.

The price-response analysis uses past-only orthogonalized estimation. There are no real price interventions, instruments, difference-in-differences episodes or randomized customer experiments in these files. Leave-venue-out evaluation and the realized-weather comparison remain unperformed. The planned minimum detectable commercial revenue effect is also unavailable. No commercial partner supplied a cost threshold; zero is used for H2, while the cost and cancellation-penalty fractions remain simulation assumptions. 

\subsection{RQ1 Booking probability forecasts}
''')
parts.append(f'''Mean validation log-loss was {cv.loc['LightGBM','mean_logloss']:.4f} for LightGBM, {cv.loc['LSTM','mean_logloss']:.4f} for the LSTM and {cv.loc['Logistic','mean_logloss']:.4f} for logistic regression. LightGBM was selected before evaluating the held-out test period. Table~\\ref{{tab:main}} reports the common test comparison. The selected model's Brier score was {f.loc['LightGBM','brier']:.4f}, compared with {f.loc['Climatology','brier']:.4f} for climatology, {f.loc['Logistic','brier']:.4f} for logistic regression, {f.loc['LSTM','brier']:.4f} for the LSTM and {f.loc['SARIMAX','brier']:.4f} for SARIMAX. Its log-loss was {f.loc['LightGBM','logloss']:.4f}. All four fitted models had positive Brier skill relative to climatology.

The selected LightGBM configuration used 240 trees, a learning rate of 0.06824, seven leaves per tree and a minimum of 80 observations per leaf. The selected LSTM had 24 hidden units and eight training epochs, with a learning rate of 0.00597. Logistic regression used $C=0.00103$. The validation search selected these configurations before their performance on the test set was known.

For LightGBM and the LSTM, the main comparison uses predictions averaged across the five model seeds. The individual LSTM runs had a mean Brier score of {seeds[seeds.model=='LSTM'].brier.mean():.4f}, with a standard deviation of {seeds[seeds.model=='LSTM'].brier.std():.4f}; their mean log-loss was {seeds[seeds.model=='LSTM'].logloss.mean():.4f}, with a standard deviation of {seeds[seeds.model=='LSTM'].logloss.std():.4f}. LightGBM's fixed, deterministic configuration produced identical probability scores across the five recorded seeds. Sampling uncertainty is reported separately through the intervals in Table~\\ref{{tab:main}}. Logistic regression was fitted once after tuning, and SARIMAX contributed one AIC-selected fit per venue.
''')
rows=[]
for name,r in f.iterrows():
    rows.append([name,f'{r.brier:.4f} [{r.brier_lo:.4f}, {r.brier_hi:.4f}]',f'{r.logloss:.4f} [{r.logloss_lo:.4f}, {r.logloss_hi:.4f}]',f'{r.auc:.4f}',f'{r.ece:.4f}',f'{r.bss:.4f}',f'{r.slope:.4f}',f'{r.intercept:.4f}'])
parts.append(table(r'Main test results on 30,660 synthetic slots. Brackets give 95\% venue-week bootstrap intervals. All values are dimensionless. Lower Brier, log-loss and ECE are preferable; larger AUC and BSS are preferable. Calibration targets are slope 1 and intercept 0.','tab:main','lccccccc',r'Model & Brier [95\% CI] & Log-loss [95\% CI] & AUC & ECE & BSS & Slope & Intercept',rows,True,r'\footnotesize'))
parts.append(f'''The paired comparisons in Table~\\ref{{tab:tests}} report the loss differences and the evidence from the planned statistical tests. LightGBM had lower Brier score and log-loss than climatology, logistic regression and SARIMAX, with Holm-adjusted $p=0.0040$ for each comparison after rounding. The differences against the LSTM were much smaller: $-0.0003$ in Brier score and $-0.0006$ in log-loss. Both paired intervals included zero, and their adjusted $p$-values were 0.1859 and 0.2369. Because the comparison with the LSTM did not establish lower loss, H1 was not satisfied.

The selected model's calibration slope was {f.loc['LightGBM','slope']:.4f}, with a 95\\% interval of {s['slope_ci'][0]:.4f} to {s['slope_ci'][1]:.4f}; its intercept was {f.loc['LightGBM','intercept']:.4f}. The slope interval excluded one, so the calibration component of H1 also failed. Figure~\\ref{{fig:reliability}} plots predicted and observed booking probabilities. The LSTM had ECE {f.loc['LSTM','ece']:.4f}, compared with {f.loc['LightGBM','ece']:.4f} for LightGBM. LightGBM remained the model selected by validation log-loss; test calibration was reported without changing that choice. All {s['sarimax_converged']} selected venue-level SARIMAX fits reported optimizer convergence.
''')
rows=[]
for _,r in tests.iterrows():rows.append([r.baseline,'Brier' if r.metric=='brier' else 'Log-loss',f'{r.difference:.4f}',f'[{r.ci_lo:.4f}, {r.ci_hi:.4f}]',f'{r.p_holm:.4f}'])
parts.append(table('Paired loss differences, LightGBM minus the named baseline. Negative values favour LightGBM. Intervals use 2,000 venue-week replicates; adjusted p-values apply Holm correction across eight tests.','tab:tests','llrrr',r'Baseline & Loss & Difference & 95\% CI & Adjusted $p$',rows,True,r'\footnotesize'))
parts.append(r'''\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{reliability.pdf}
\caption{Test-set reliability on synthetic slots. Both axes are dimensionless probabilities. Points show ten equal-width probability bins with at least 20 observations; the dashed diagonal indicates equal predicted and observed booking rates.}
\label{fig:reliability}
\end{figure}
''')
parts.append(r'''\begin{figure*}[t]
\centering\includegraphics[width=.98\textwidth]{elasticity_sensitivity.pdf}
\caption{Price-response diagnostics on synthetic data. Left: estimated segment probability derivatives with 95\% clustered intervals and simulator average local derivatives; the segment index follows the saved elasticity table. Right: revenue change for the fixed chosen policy when estimated elasticities are multiplied by 0.5--1.5. The horizontal dashed line gives the known simulator effect for that policy.}
\label{fig:elasticity}
\end{figure*}
''')
parts.append(f'''At the venue-day level, the LSTM had MAE {f.loc['LSTM','mae_bookings_day']:.3f} bookings per day and WAPE {f.loc['LSTM','wape_percent']:.2f}\\%, compared with {f.loc['LightGBM','mae_bookings_day']:.3f} and {f.loc['LightGBM','wape_percent']:.2f}\\% for LightGBM. Table~\\ref{{tab:secondary}} gives the remaining values. These aggregated measures were not assessed with a separate paired test, so the reported ordering is descriptive. Aggregation changes the scored quantity from individual booking probabilities to the expected number of bookings across each venue-day.

\\subsection{{RQ2 Price response and counterfactual revenue}}
Cross-fitting supplied {s['crossfit_records']:,} residual observations across 32 segments. Segment counts ranged from {int(el.n.min()):,} to {int(el.n.max()):,}. The residual-to-observed log-price variance ratio ranged from {s['residual_share_min']:.3f} to {s['residual_share_max']:.3f}, and all {s['n_used_segments']} segments met the sample-size and residual-variance thresholds. Five estimated probability derivatives were positive, although the simulator's local booking response to price was negative. Figure~\\ref{{fig:elasticity}} shows the estimates and their intervals in the segment order retained in the accompanying CSV.

Against the simulator's average local derivative, the mean absolute discrepancy was {s['theta_mae']:.4f} for the orthogonalized estimate and {s['naive_mae']:.4f} for the naive estimate. The simulator derivative lay inside 29 of the 32 estimated intervals, or {s['elasticity_coverage']*100:.1f}\\%. This check compares the fitted partially linear coefficient with the average local derivative of the nonlinear generator. The two quantities have different statistical targets, so interval overlap is reported as a diagnostic.

The training revenue normalization factor was {s['calibration_factor']:.4f}. On test data, modeled revenue at historical prices divided by observed booking revenue was {s['historical_revenue_diagnostic']:.4f}. The corresponding modeled booking rate was {s['historical_demand_rate']*100:.2f}\\%, compared with the observed 32.58\\%. The policy changed prices for {s['changed_share']*100:.2f}\\% of slots, and the mean recommended-to-historical price ratio was {s['mean_price_ratio']:.4f}. There were 20,259 price increases, 9,655 decreases and 746 unchanged prices, counting differences smaller than one millionth of a currency unit as unchanged. The no-change option remained available in the optimization. These recommendations were evaluated offline and were not implemented with customers. The demand diagnostics use the fitted price-response layer; checks using the generator's hidden booking probabilities are reported separately.

Table~\\ref{{tab:policy}} reports a modeled revenue change of {pol.loc['revenue_percent','estimate']:.2f}\\%, with a 95\\% interval of {ci(pol.loc['revenue_percent'])}\\%. Its lower bound exceeded the zero threshold, so the estimated-demand calculation satisfied H2. Evaluating exactly the same recommended prices under the simulator's known booking function gave {tru.loc['revenue_percent','truth']:.2f}\\%. That value was below the lower endpoint of the estimated interval. The absolute difference between estimated and simulator revenue changes was {pol.loc['revenue_percent','estimate']-tru.loc['revenue_percent','truth']:.2f} percentage points. No realized commercial revenue increase was observed.
''')
rows=[]
labels={'revenue_percent':r'Revenue change (\%)','utilization_pp':'Utilization change (pp)','offpeak_pp':'Off-peak change (pp)','cancellation_pp':'Cancellation change (pp)'}
for key,r in pol.iterrows():rows.append([labels[key],f'{r.estimate:.2f}',f'[{r.lo:.2f}, {r.hi:.2f}]',f'{tru.loc[key,"truth"]:.2f}'])
parts.append(table('Recommended minus historical policy on the same synthetic test slots. Estimated intervals combine venue-week resampling and 1,000 elasticity draws. The simulator column evaluates the fixed chosen prices using the known mechanism and is a deterministic expectation, not observed customer revenue.','tab:policy','lrrr',r'Outcome & Estimate & 95\% CI & Simulator',rows,size=r'\footnotesize'))
parts.append(f'''With the chosen policy held fixed, multiplying all estimated elasticities by a common factor from 0.5 to 1.5 produced revenue changes from {sens.revenue_percent.min():.2f}\\% to {sens.revenue_percent.max():.2f}\\%. Figure~\\ref{{fig:elasticity}} shows the complete sensitivity curve. The calculated changes remained positive across the tested factors; no break-even factor was observed in this interval. The horizontal simulator value evaluates the same policy under its full known response, whereas the curve changes one shared multiplier in the estimated response model.

\\subsection{{RQ3 Off-peak occupancy and cancellations}}
Under estimated demand, overall utilization changed by {pol.loc['utilization_pp','estimate']:.2f} percentage points, with an interval of {ci(pol.loc['utilization_pp'])}. The simulator evaluation gave {tru.loc['utilization_pp','truth']:.2f} percentage points for the same prices. The estimated response predicted an increase, while the simulator response gave a decrease in overall utilization. Off-peak occupancy changed by {pol.loc['offpeak_pp','estimate']:.2f} percentage points under estimated demand, with an interval of {ci(pol.loc['offpeak_pp'])}; the corresponding simulator change was {tru.loc['offpeak_pp','truth']:.2f} percentage points.

The estimated cancellation-rate change was {pol.loc['cancellation_pp','estimate']:.2f} percentage points, with an interval of {ci(pol.loc['cancellation_pp'])}. The training-derived non-inferiority margin was {s['cancellation_margin_pp']:.2f} percentage points, and the estimated upper bound was below that margin. The cancellation condition passed, but the lower bound for off-peak occupancy did not exceed zero. H3 was therefore not satisfied. Both cancellation rates use expected bookings as the denominator. Offered slots without a booking are excluded from that denominator.
''')

parts.append(f'''\\subsection{{Additional analyses and observed errors}}
Removing the four weather variables from every covariate step increased LightGBM's Brier score from {ab.loc['brier','with_weather']:.4f} to {ab.loc['brier','without_weather']:.4f} and log-loss from {ab.loc['logloss','with_weather']:.4f} to {ab.loc['logloss','without_weather']:.4f}. The paired Brier difference was {ab.loc['brier','difference']:.4f}, with a 95\\% interval of {ci(ab.loc['brier'],4)}. The corresponding log-loss interval was {ci(ab.loc['logloss'],4)}. The ablation was evaluated separately from the eight primary tests. Its bootstrap intervals are descriptive, and no multiplicity-adjusted significance is claimed for these additional comparisons.

Figure~\\ref{{fig:groups}} compares observed and predicted booking rates by facility type. Football had an observed rate of {g.loc['football','observed']*100:.2f}\\% and a predicted rate of {g.loc['football','mean_pred']*100:.2f}\\%, the largest absolute rate discrepancy among the four types. Swimming had the highest group Brier score, {g.loc['swimming','brier']:.4f}; tennis had the lowest, {g.loc['tennis','brier']:.4f}. Booking prevalence differs across these groups. The scores describe each group's prediction errors without establishing a controlled comparison of facility difficulty. The saved group table includes denominators, booked counts and log-loss for every type.

A further descriptive error check examined the 1,875 test slots with predicted booking probability of at least 0.60. Their mean prediction was 65.74\\%, and 1,280 were booked, an observed rate of 68.27\\%. The remaining 595 slots were unbooked despite receiving predictions above that threshold. The 0.60 threshold was used for this descriptive error check only. It had no role in model selection or recalibration.

The seasonal breakdown contains 11,340 spring slots and 19,320 summer slots. LightGBM's Brier score was {g.loc['spring','brier']:.4f} in spring and {g.loc['summer','brier']:.4f} in summer; the observed booking rates were {g.loc['spring','observed']*100:.2f}\\% and {g.loc['summer','observed']*100:.2f}\\%. No autumn or winter test scores were available. The test period therefore provides two seasonal groups, while all four seasons occur in the development data.

Several parts of the pipeline did not meet the specified checks. LightGBM's loss advantage over the LSTM was not supported, five estimated price derivatives were positive, and the simulator revenue effect fell outside the estimated interval. Estimated and simulator utilization changes also had opposite signs. H1 and H3 did not satisfy all their conditions. The selected model, price policy, test set and success thresholds were kept fixed after these outcomes were observed. Their causes and implications are left for the Discussion.
''')
rows=[]
for name,r in f.iterrows():rows.append([name,f'{r.mae_bookings_day:.3f}',f'{r.wape_percent:.2f}'])
parts.append(table('Venue-day errors after summing slot probabilities. MAE is measured in bookings per venue-day and WAPE is a percentage. These secondary measures use the same test slots as the main probability comparison.','tab:secondary','lrr',r'Model & MAE (bookings/day) & WAPE (\%)',rows))
parts.append(r'''\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{group_errors.pdf}
\caption{Observed and LightGBM-predicted booking rates by synthetic facility type on the common test period. Football and futsal each have 8,760 slots; swimming and tennis each have 6,570. Bars show rates as percentages of offered slots.}
\label{fig:groups}
\end{figure}
''')
v=s['versions']
parts.append(f'''\\subsection{{Reproducibility}}
\\input{{repository}}
The run used Python {s['python']}, NumPy {v['numpy']}, pandas {v['pandas']}, scikit-learn {v['sklearn']}, SciPy {v['scipy']}, LightGBM {v['lightgbm']}, Optuna {v['optuna']}, PyTorch {v['torch'].replace('+',r'{+}')} and statsmodels {v['statsmodels']}. The execution platform was Linux on an {s['cpu_model']}, with {s['cpu_count']} logical CPUs visible and two computation threads configured for the fitted models. No GPU was used. A clean analysis run took approximately {s['runtime_seconds']/60:.1f} minutes in this environment, excluding package installation and manuscript compilation; runtime on another computer will differ.

The generator and tuning seeds were 2026, model seeds were 0--4, the forecast bootstrap seed was 2026, and the combined elasticity and policy bootstrap used seed 123. The package records split boundaries, search trials, selected hyperparameters, per-seed scores, test predictions, paired tests, elasticity estimates and simulator checks. It includes a script that checks reported scores against saved predictions and verifies the test membership, temporal gaps, price grid and revenue arithmetic. All tabular values and vector figures in this section can be regenerated from the supplied data and code.
\\FloatBarrier
''')
(P/'results.tex').write_text('\n\n'.join(parts))
print('Results source written; whitespace word count including LaTeX:',len('\n'.join(parts).split()))
