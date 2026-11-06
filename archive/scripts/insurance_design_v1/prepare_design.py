from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).parent
plans=[]
for name,price,cd,cl in [('Basic',3,50,300),('Standard',6,100,600),('Premium',11,200,1200)]:
 plans.append({'name':name,'price':price,'delay_benefit':cd,'loss_benefit':cl,'reference':{'price':price,'delay_benefit':cd,'loss_benefit':cl}})
config={'status':'proposed_v1_not_calibrated_not_final_quote','reference_candidate_source':'user-pasted ChatGPT design; illustrative values only',
 'policy_proposal':{'unit':'one named passenger, one checked bag, one direct one-way flight','scope':'Hong Kong departure pilot proposed, not imposed on BTS training population','delay_threshold_hours':12,'delay_clock':'passenger actual arrival at covered destination to verified availability for collection','final_loss_confirmation':'carrier confirmation or explicit policy recognition deadline; deadline not yet selected','settlement':'delay advance then loss top-up; final total capped at loss benefit','no_change_to_existing_carrier_liability':True},
 'plans':plans,
 'assumptions':{'currency':'USD','covered_bags':1,'covered_segments':1,'eligible_opportunity_fraction':None,
 'conditional_delay_share':None,'conditional_loss_share':None,
 'tier_hazard_multipliers':{p['name']:None for p in plans},
 'risk_states':[{'name':'normal','probability':None,'hazard_multiplier':None},{'name':'shared_disruption','probability':None,'hazard_multiplier':None}],
 'demand':{'reference_purchase_probabilities':{p['name']:None for p in plans},'log_price_sensitivity':None,'log_delay_benefit_sensitivity':None,'log_loss_benefit_sensitivity':None},
 'variable_cost_per_policy':{p['name']:None for p in plans},'external_expense_rates':{p['name']:None for p in plans},
 'period_system_cost':None,'target_period_profit':None},
 'finance_scope':'whole proposed product initially; external expenses only; internal commission cancels at consolidated scope',
 'optimization':{'objectives':['expected_period_profit','expected_period_profit_margin'],'required_constraints':['explicit minimum delay/loss benefits','expected profit >= chosen target after system costs','loss benefit >= delay benefit','sales mutually exclusive by tier','chosen downside-risk tolerance'],'probability_profit_target':None,'claim_frequency_cap':None,'payout_to_premium_cap':None},
 'risk_model':'frozen comprehensive forecasting pipeline, unchanged by this design',
 'time_scope':'latest-month cohort; annual analysis requires a full monthly exposure/risk scenario sequence, not August repeated silently'}
(ROOT/'design_config_template.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'candidate_tiers_reference_only.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['plan','currency','candidate_price','delay_benefit','loss_benefit','status']);w.writeheader()
 for p in plans:w.writerow({'plan':p['name'],'currency':'USD','candidate_price':p['price'],'delay_benefit':p['delay_benefit'],'loss_benefit':p['loss_benefit'],'status':'reference candidate; not approved quote'})
source=ROOT.parent/'latest_bts_pipeline_test/predictions.csv'
rows=[]
for r in csv.DictReader(source.open(encoding='utf-8')):
 if r['target_month']=='2026-08-01':
  rows.append({'period':r['target_month'],'carrier':r['carrier'],'risk_rate_per_1000':float(r['rate_ml']),'passenger_segments':float(r['forecast_passenger_segments']),'bag_loads':float(r['forecast_baggage_volume']),'predicted_mishandled_bags':float(r['anomaly_count_ml']),'training_cutoff':r['training_last_month'],'population':'BTS US domestic, not Cathay','forecast_status':'retrospective August forecast; no actual August labels in interface'})
assert len(rows)==12 and all(r['training_cutoff']=='2026-07-01' for r in rows)
with (ROOT/'forecast_interface_2026_08.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
missing=[]
def collect(x,path='assumptions'):
 if x is None:missing.append(path)
 elif isinstance(x,dict):
  for k,v in x.items():collect(v,path+'.'+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):collect(v,f'{path}[{i}]')
collect(config['assumptions'])
(ROOT/'readiness.json').write_text(json.dumps({'design_complete':True,'model_implementation_present':True,'calibrated_financial_evaluation_ready':False,'unset_assumptions':missing,'forecast_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'forecasts_imported':len(rows),'existing_forecast_files_modified':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared candidate tiers, config, 12 forecast inputs; unknown parameters kept null.')
