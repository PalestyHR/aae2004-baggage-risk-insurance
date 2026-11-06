"""Budget-first feasibility model. No default benefit or claim conversion."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/baggage_ml_dependencies'))
import pandas as pd
OUT=Path(__file__).with_suffix('');OUT.mkdir(exist_ok=True)
def payout_budget(*,eligible_opportunities,purchase_rate,premium,nonclaim_costs,target_profit):
 if eligible_opportunities<0 or not 0<=purchase_rate<=1 or min(premium,nonclaim_costs,target_profit)<0:raise ValueError('Invalid financial inputs')
 policies=eligible_opportunities*purchase_rate
 revenue=policies*premium
 budget=revenue-nonclaim_costs-target_profit
 return {'expected_policies':policies,'premium_revenue':revenue,'available_expected_payout_budget':budget,'feasible_before_claims':budget>=0}
def allowable_average_benefit(*,expected_covered_claims,available_expected_payout_budget):
 if expected_covered_claims<=0:raise ValueError('Positive expected covered claims required; zero does not imply unlimited safe benefit')
 if available_expected_payout_budget<0:return None
 return available_expected_payout_budget/expected_covered_claims
def build():
 d=pd.read_csv(ROOT/'Assignments/Group_Project/Material/integrated_dataset/dataset/carrier_month_observed_NOT_model_input.csv')
 rows=[]
 for keys,g in d.groupby(['year','month']):
  matched=g[g.passenger_segments.notna()]
  rows.append({'year':int(keys[0]),'month':int(keys[1]),'baggage_reporting_carriers':g.carrier.nunique(),'matched_passenger_carriers':matched.carrier.nunique(),'passenger_segments_matched':matched.passenger_segments.sum(min_count=1),'mishandled_bags_matched':matched.mishandled_bags.sum(min_count=1),'enplaned_bags_matched':matched.enplaned_bags.sum(min_count=1),'mishandled_bags_all_available':g.mishandled_bags.sum(),'enplaned_bags_all_available':g.enplaned_bags.sum(),'permanently_lost_bags':None,'lost_count_status':'not separately reported','scope':'US domestic reported carrier-month subset, not Cathay or entire market'})
 monthly=pd.DataFrame(rows);monthly.to_csv(OUT/'monthly_observed_volumes.csv',index=False)
 recent=monthly[(monthly.year==2026)&(monthly.month<=7)]
 totals={'period':'2026-01 through 2026-07','passenger_segments':int(recent.passenger_segments_matched.sum()),'mishandled_bags':int(recent.mishandled_bags_matched.sum()),'enplaned_bags':int(recent.enplaned_bags_matched.sum()),'permanently_lost_bags':None,'cathay_passengers_carried_july2026':2714415,'cathay_passengers_carried_jan_july2026':18720063,'cathay_mishandled_bags':None,'cathay_permanently_lost_bags':None,'cathay_source':'https://news.cathaypacific.com/the-cathay-group-releases-traffic-figures-for-july-2026','cathay_source_publication_date':'2026-08-19','caveat':'Do not multiply US bag-rate by Cathay passenger count. Bag exposure and portfolio calibration are missing.'}
 (OUT/'volume_summary.json').write_text(json.dumps(totals,indent=2),encoding='utf-8')
 plan={'purpose':'design compensation only after finding feasible revenue-cost budget','variables':{'V':'reported passenger traffic; not automatically eligible policy opportunities','s':'conversion from passenger traffic to eligible insured itinerary opportunities, unknown','q':'purchase rate, unknown','P':'premium to explore, not fixed','K':'all nonclaim costs on chosen business scope, unknown','G':'target profit, to set','C':'expected valid claims among insured policies, unknown','B':'average benefit to design'},'equations':{'eligible_opportunities':'V*s','sales':'V*s*q','revenue':'V*s*q*P','expected_payout_budget':'V*s*q*P-K-G','average_benefit_upper_bound':'(V*s*q*P-K-G)/C for C>0 and nonnegative budget','multi_tier_constraint':'sum_k C_k*B_k <= total_revenue-K-G'},'not_estimated':['s','q','K','G','C','permanent_loss_rate','Cathay_baggage_exposure'],'status':'expected-value feasibility only; not a solvency or guaranteed-profit result','previous_scenario':'50% qualification and HKD1000 benefit withdrawn as product inputs; previous outputs illustrative only'}
 (OUT/'design_parameters.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 # Checks use arbitrary numeric fixtures, not saved product parameters.
 x=payout_budget(eligible_opportunities=100,purchase_rate=.2,premium=10,nonclaim_costs=30,target_profit=20)
 assert x['available_expected_payout_budget']==150
 assert allowable_average_benefit(expected_covered_claims=3,available_expected_payout_budget=150)==50
 assert int(recent.mishandled_bags_matched.sum())==1273961
 print(json.dumps(totals,indent=2));print('PASS volume reconciliation and symbolic-budget implementation')
if __name__=='__main__':build()
