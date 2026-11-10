"""Same conservative pointwise ratio convention as the earlier W1/Wc report."""
from scipy.stats import beta
import math

def arm_interval(errors,shots,coverage=.975):
    if not 0<=errors<=shots or shots<1:raise ValueError('Need 0 <= errors <= shots and positive shots')
    a=(1-coverage)/2
    return (0.0 if errors==0 else float(beta.ppf(a,errors,shots-errors+1)),
            1.0 if errors==shots else float(beta.ppf(1-a,errors+1,shots-errors)))

def ratio_interval(candidate_errors,candidate_shots,baseline_errors,baseline_shots):
    a=arm_interval(candidate_errors,candidate_shots);b=arm_interval(baseline_errors,baseline_shots)
    return dict(candidate_rate=candidate_errors/candidate_shots,baseline_rate=baseline_errors/baseline_shots,
                ratio=(candidate_errors/candidate_shots)/(baseline_errors/baseline_shots) if baseline_errors else None,
                lower=a[0]/b[1],upper=a[1]/b[0] if b[0] else None,
                upper_unbounded=(b[0]==0),pointwise_coverage_at_least=.95,
                method='97.5% Clopper-Pearson per arm, ratio endpoints, Bonferroni',
                interval_crosses_one_means_equivalence=False)

if __name__=='__main__':
    r=ratio_interval(2500,1000000,2500,1000000)
    assert r['lower']<1<r['upper'] and r['upper']<1.11
    assert ratio_interval(0,1000,0,1000)['upper_unbounded']
    print(r)
