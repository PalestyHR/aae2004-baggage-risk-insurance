"""Insurance design layer on the frozen BTS forecasting pipeline.

All monetary values use one declared currency. No probabilities, customer
preferences or system costs are silently calibrated from BTS mishandling data.
The portfolio unit here is one sales cohort, e.g. one month's departures.
"""
import math

def bounded(x, name, lo=0.0, hi=1.0):
    if x is None or not math.isfinite(float(x)) or not lo <= x <= hi:
        raise ValueError(f'{name} must be supplied in [{lo}, {hi}]')
    return float(x)

def nonnegative(x, name):
    if x is None or not math.isfinite(float(x)) or x < 0:
        raise ValueError(f'{name} must be supplied and nonnegative')
    return float(x)

def positive(x, name):
    x=nonnegative(x,name)
    if x == 0: raise ValueError(f'{name} must be positive')
    return x

def claims_from_forecast(rate_per_1000, bag_count, segment_count,
                         portfolio_hazard_multiplier, delay_share, loss_share):
    """Assumed unit mapping, NOT empirical calibration.
    One bag/one segment reduces to the scaled baggage rate at multiplier=1.
    Additional exposures use an independence-style survival approximation.
    Shares are conditional FINAL policy outcomes, not bag-level industry shares.
    """
    p=bounded(rate_per_1000/1000,'predicted rate / 1000')
    for name,x in [('bag_count',bag_count),('segment_count',segment_count)]:
        if not isinstance(x,int) or isinstance(x,bool) or x<1: raise ValueError(name+' must be a positive integer')
    rho=nonnegative(portfolio_hazard_multiplier,'portfolio_hazard_multiplier')
    sd=bounded(delay_share,'conditional delay share');sl=bounded(loss_share,'conditional loss share')
    if sd+sl>1+1e-12: raise ValueError('Final delay and loss shares must sum to <=1')
    exposures=bag_count*segment_count
    qm=0.0 if rho==0 or p==0 else (1.0 if p==1 else -math.expm1(exposures*rho*math.log1p(-p)))
    qd,ql=qm*sd,qm*sl
    return {'policy_mishandling_probability':qm,'delay_claim_probability':qd,
            'loss_claim_probability':ql,'no_paid_claim_probability':max(0.,1-qd-ql),
            'status':'scenario_conversion_not_calibrated'}

def tier_probabilities(plans, demand):
    """Multinomial logit with a no-purchase option; a customer buys <=1 tier.
    Reference shares and preference coefficients are required scenario inputs.
    Coverage changes affect utility explicitly, not just cost.
    """
    names=[p['name'] for p in plans]
    if len(set(names))!=len(names): raise ValueError('Duplicate plan names')
    shares=demand['reference_purchase_probabilities']
    if set(shares)!=set(names): raise ValueError('Reference shares must match plan names')
    total=sum(bounded(shares[n],n+' reference share') for n in names)
    if total>=1: raise ValueError('Positive reference no-purchase share required')
    q0=1-total;logits=[0.]
    beta=nonnegative(demand['log_price_sensitivity'],'log_price_sensitivity')
    gd=nonnegative(demand['log_delay_benefit_sensitivity'],'log_delay_benefit_sensitivity')
    gl=nonnegative(demand['log_loss_benefit_sensitivity'],'log_loss_benefit_sensitivity')
    for p in plans:
        ref=p['reference']
        price=positive(p['price'],'price');cd=positive(p['delay_benefit'],'delay_benefit');cl=positive(p['loss_benefit'],'loss_benefit')
        if cl<cd: raise ValueError('Loss benefit must cover previously paid delay benefit')
        utility=-math.inf if shares[p['name']]==0 else math.log(shares[p['name']]/q0)
        utility-=beta*math.log(price/positive(ref['price'],'reference price'))
        utility+=gd*math.log(cd/positive(ref['delay_benefit'],'reference delay benefit'))
        utility+=gl*math.log(cl/positive(ref['loss_benefit'],'reference loss benefit'))
        logits.append(utility)
    peak=max(logits);exps=[math.exp(x-peak) for x in logits];den=sum(exps)
    return dict(zip(['no_purchase']+names,[x/den for x in exps]))

def expected_portfolio(forecast,plans,assumptions):
    """Returns expected cohort economics, after specified allocated system cost.
    No forecasting test labels are accepted or needed by this interface.
    """
    names=[p['name'] for p in plans]
    V=nonnegative(forecast['passenger_segments'],'forecast passenger segments')
    B=nonnegative(forecast['bag_loads'],'forecast bag loads')
    s=bounded(assumptions['eligible_opportunity_fraction'],'eligible_opportunity_fraction')
    n=math.floor(V*s)
    bags=assumptions['covered_bags'];segments=assumptions['covered_segments']
    if n*bags*segments>B+1e-6: raise ValueError('Eligible policy exposures exceed forecast bag loads; revise unit mapping')
    probs=tier_probabilities(plans,assumptions['demand'])
    states=assumptions['risk_states']
    if not states or not math.isclose(sum(bounded(t['probability'],'state probability') for t in states),1,abs_tol=1e-9):
        raise ValueError('Risk-state probabilities must sum to one')
    if len({t['name'] for t in states})!=len(states):raise ValueError('Duplicate risk-state names')
    sd=assumptions['conditional_delay_share'];sl=assumptions['conditional_loss_share']
    K=nonnegative(assumptions['period_system_cost'],'period_system_cost')
    G=nonnegative(assumptions['target_period_profit'],'target_period_profit')
    if set(assumptions['tier_hazard_multipliers'])!=set(names):raise ValueError('Provide each tier hazard multiplier')
    if set(assumptions['variable_cost_per_policy'])!=set(names):raise ValueError('Provide each tier variable cost')
    if set(assumptions['external_expense_rates'])!=set(names):raise ValueError('Provide each tier expense rate')
    tier_rows=[];state_probs={}
    for p in plans:
        name=p['name'];qd=ql=0;state_probs[name]={}
        for state in states:
            rho=nonnegative(assumptions['tier_hazard_multipliers'][name],'tier multiplier')*nonnegative(state['hazard_multiplier'],'state multiplier')
            risk=claims_from_forecast(forecast['risk_rate_per_1000'],bags,segments,rho,sd,sl)
            state_probs[name][state['name']]=risk
            qd+=state['probability']*risk['delay_claim_probability'];ql+=state['probability']*risk['loss_claim_probability']
        nk=n*probs[name];mu=qd*p['delay_benefit']+ql*p['loss_benefit']
        v=nonnegative(assumptions['variable_cost_per_policy'][name],'variable cost')
        e=bounded(assumptions['external_expense_rates'][name],'external expense rate',0,.999999999)
        rev=nk*p['price'];claim=nk*mu;expense=nk*v+e*rev
        tier_rows.append({'name':name,'purchase_probability':probs[name],'expected_policies':nk,
                         'delay_claim_probability':qd,'loss_claim_probability':ql,
                         'expected_paid_policies':nk*(qd+ql),'expected_claim_per_policy':mu,
                         'revenue':rev,'claims':claim,'nonclaim_variable_expense':expense,
                         'contribution_before_shared_system_cost':rev-claim-expense})
    rev=sum(r['revenue'] for r in tier_rows);claim=sum(r['claims'] for r in tier_rows)
    expense=sum(r['nonclaim_variable_expense'] for r in tier_rows);sales=sum(r['expected_policies'] for r in tier_rows)
    profit=rev-claim-expense-K;budget=rev-expense-K-G
    return {'currency':assumptions['currency'],'forecast_period':forecast['period'],
            'eligible_opportunities':n,'tiers':tier_rows,'choice_probabilities':probs,
            'risk_states':states,'state_claim_probabilities':state_probs,
            'expected_sales':sales,'expected_revenue':rev,'expected_claims':claim,
            'nonclaim_variable_expense':expense,'period_system_cost':K,
            'expected_profit':profit,'expected_profit_margin':profit/rev if rev else None,
            'expected_loss_ratio':claim/rev if rev else None,
            'expected_paid_policy_fraction':sum(r['expected_paid_policies'] for r in tier_rows)/sales if sales else None,
            'allowable_total_payout_at_target_profit':budget,
            'payout_budget_slack':budget-claim,'meets_target_expected_profit':profit>=G,
            'evidence_status':'scenario_economics_not_validated_sales_or_claims'}

def benefit_frontier(*,policies,price,expense_rate,variable_cost,allocated_fixed_cost,
                     target_profit,qd,ql,delay_benefit):
    """Conditional loss-benefit ceiling. If price/benefit changes uptake, rerun demand.
    Returns no automatic infinite benefit when loss probability is zero.
    """
    N=positive(policies,'policies');P=positive(price,'price');e=bounded(expense_rate,'expense rate',0,.999999999)
    v=nonnegative(variable_cost,'variable cost');K=nonnegative(allocated_fixed_cost,'fixed cost');G=nonnegative(target_profit,'target profit')
    qd=bounded(qd,'qd');ql=bounded(ql,'ql');cd=nonnegative(delay_benefit,'delay benefit')
    if qd+ql>1:raise ValueError('Invalid mutually exclusive probabilities')
    allowance=P*(1-e)-v-(K+G)/N
    residual=allowance-qd*cd
    return {'payout_allowance_per_policy':allowance,'remaining_after_delay':residual,
            'loss_benefit_ceiling':residual/ql if ql>0 and residual>=0 else None,
            'status':'conditional_ceiling' if ql>0 and residual>=0 else 'infeasible_or_unidentified'}

def simulate_cohort(forecast,plans,assumptions,draws=20000,seed=42):
    """Scenario-only cohort profit distribution. A shared risk state creates
    cross-policy dependence; conditional claims use mutually exclusive multinomial
    outcomes. Not an annual probability from one observed month.
    """
    import numpy as np
    if not isinstance(draws,int) or draws<100:raise ValueError('Use >=100 draws')
    expected=expected_portfolio(forecast,plans,assumptions);rng=np.random.default_rng(seed)
    names=[p['name'] for p in plans];choice=expected['choice_probabilities']
    counts=rng.multinomial(expected['eligible_opportunities'],[choice[n] for n in ['no_purchase']+names],size=draws)
    states=assumptions['risk_states'];stateidx=rng.choice(len(states),size=draws,p=[x['probability'] for x in states])
    profits=np.full(draws,-float(assumptions['period_system_cost']))
    for k,p in enumerate(plans,start=1):
        nk=counts[:,k];name=p['name'];price=p['price']
        profits+=nk*(price*(1-assumptions['external_expense_rates'][name])-assumptions['variable_cost_per_policy'][name])
        for j,state in enumerate(states):
            inds=np.where(stateidx==j)[0]
            risk=expected['state_claim_probabilities'][name][state['name']]
            qd=risk['delay_claim_probability'];ql=risk['loss_claim_probability']
            # Vectorized sequential binomials are equivalent to a multinomial:
            # lost first, then delayed among policies not classed as lost.
            lost=rng.binomial(nk[inds],ql)
            delayed=rng.binomial(nk[inds]-lost,qd/(1-ql) if ql<1 else 0)
            profits[inds]-=lost*p['loss_benefit']+delayed*p['delay_benefit']
    return {'expected_profit_analytic':expected['expected_profit'],
            'simulated_mean_profit':float(profits.mean()),'mean_standard_error':float(profits.std(ddof=1)/math.sqrt(draws)),
            'probability_profit_positive':float((profits>0).mean()),
            'probability_target_met':float((profits>=assumptions['target_period_profit']).mean()),
            'profit_quantile_05':float(np.quantile(profits,.05)),
            'profit_quantile_50':float(np.quantile(profits,.5)),
            'profit_quantile_95':float(np.quantile(profits,.95)),
            'draws':draws,'seed':seed,'scope':'single cohort under explicit assumptions; not calibrated annual profitability'}

def pareto_frontier(candidates):
    """Maximize expected profit and margin, among externally constrained designs.
    Customer coverage/demand/risk requirements must be checked before passing in.
    """
    feasible=[c for c in candidates if c.get('feasible',False) and c.get('expected_profit_margin') is not None]
    result=[]
    for a in feasible:
        if not all(math.isfinite(a[x]) for x in ['expected_profit','expected_profit_margin']):raise ValueError('Nonfinite objective')
        dominated=any(b['expected_profit']>=a['expected_profit'] and b['expected_profit_margin']>=a['expected_profit_margin'] and
                      (b['expected_profit']>a['expected_profit'] or b['expected_profit_margin']>a['expected_profit_margin']) for b in feasible)
        if not dominated:result.append(a)
    return result
