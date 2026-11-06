"""Exploratory comparison. Reuses exact audited data/feature definitions.
Select alpha on pre-2025 rolling validation, model family on 2025 only.
2026 is already inspected in earlier work and is not an untouched final test.
"""
from pathlib import Path
# Load only shared definitions, without rerunning the previous experiment.
source=Path(__file__).with_name('periodic_learning.py').read_text(encoding='utf-8')
exec(compile(source.split('predictions=[];validation=[];summary=[];audit=[]')[0],str(Path(__file__).with_name('periodic_learning.py')),'exec'))
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import lightgbm,xgboost,sklearn
OUT=Path(__file__).parent/'boosting_comparison'
OUT.mkdir(exist_ok=True)
class Candidate:
 def __init__(self,name,alpha): self.name=name;self.alpha=alpha
 def fit(self,frame):
  self.prep=ColumnTransformer([('id',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['AIRLINE_ID']),('num',StandardScaler(),features[1:])])
  X=self.prep.fit_transform(frame[features]);y=frame.rate.to_numpy()
  self.linear=None;self.tree=None
  if 'lasso' in self.name:
   self.linear=Lasso(alpha=self.alpha,max_iter=20000,tol=1e-6).fit(X,y)
   residual=y-self.linear.predict(X)
  else: residual=y
  if 'lightgbm' in self.name:
   self.tree=LGBMRegressor(n_estimators=200,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=20,reg_lambda=1.0,verbosity=-1,n_jobs=2,random_state=42)
  if 'xgboost' in self.name:
   self.tree=XGBRegressor(n_estimators=200,learning_rate=.03,max_depth=3,min_child_weight=20,reg_lambda=1.0,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=42)
  if self.tree is not None: self.tree.fit(X,residual)
  return self
 def predict(self,frame):
  X=self.prep.transform(frame[features]);p=np.zeros(len(frame))
  if self.linear is not None: p+=self.linear.predict(X)
  if self.tree is not None: p+=self.tree.predict(X)
  # ReLU-like final nonnegativity constraint, NOT a neural activation layer.
  return np.maximum(p,0)
def run(data,year,delay,name,alpha,mode,last_month=12):
 result=[];fitted=None;initial=year*12-delay
 if mode=='fixed': fitted=Candidate(name,alpha).fit(training(data,initial,'expanding'))
 for t in sorted(data.loc[(data.YEAR==year)&(data.MONTH<=last_month),'t'].unique()):
  test=data[data.t==t];cutoff=t-delay
  if mode=='updating': fitted=Candidate(name,alpha).fit(training(data,cutoff,'expanding'))
  o=test[['YEAR','MONTH','t','AIRLINE_ID','CARRIER','rate','ENPLANED_BAGGAGE']].copy()
  o['prediction']=fitted.predict(test);o['model']=name;o['mode']=mode;o['delay']=delay;o['alpha']=alpha
  o['fit_cutoff']=initial if mode=='fixed' else cutoff
  assert o.fit_cutoff.le(o.t-delay).all()
  result.append(o)
 return pd.concat(result,ignore_index=True)
names=['lasso','lightgbm','xgboost','lasso_lightgbm','lasso_xgboost']
alpha_scores=[];validation=[];summaries=[];predictions=[];choices=[]
for delay in [1,3]:
 data=build(delay)
 # Tune Lasso on 2024 only, within a modest predeclared grid.
 for alpha in [.001,.01,.1]:
  p=run(data,2024,delay,'lasso',alpha,'updating',last_month=13-delay)
  alpha_scores.append({'delay':delay,'alpha':alpha,**metrics(p)})
 alpha=min([s for s in alpha_scores if s['delay']==delay],key=lambda s:s['weighted_MAE'])['alpha']
 for name in names:
  for mode in ['fixed','updating']:
   p=run(data,2025,delay,name,alpha,mode,last_month=13-delay)
   validation.append({'delay':delay,'model':name,'mode':mode,'alpha':alpha,**metrics(p)})
 best=min([s for s in validation if s['delay']==delay],key=lambda s:s['weighted_MAE'])
 choices.append({'delay':delay,'alpha':alpha,'model':best['model'],'mode':best['mode']})
 print('Selected before 2026:',choices[-1],flush=True)
 for name in names:
  for mode in ['fixed','updating']:
   p=run(data,2026,delay,name,alpha,mode,last_month=8);predictions.append(p)
   summaries.append({'delay':delay,'model':name,'mode':mode,'alpha':alpha,**metrics(p)})
allp=pd.concat(predictions,ignore_index=True)
monthly=[]
for keys,g in allp.groupby(['delay','model','mode','MONTH']): monthly.append(dict(zip(['delay','model','mode','MONTH'],keys),**metrics(g)))
for name,rows in [('alpha_validation_2024',alpha_scores),('validation_2025',validation),('summary_2026',summaries),('monthly_2026',monthly)]: pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
allp.to_csv(OUT/'predictions_2026.csv',index=False)
meta={'choices_before_2026':choices,'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'versions':{'lightgbm':lightgbm.__version__,'xgboost':xgboost.__version__,'sklearn':sklearn.__version__},'results':summaries}
(OUT/'audit.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(pd.DataFrame(summaries)[['delay','model','mode','weighted_MAE','RMSE']].to_string(index=False))
