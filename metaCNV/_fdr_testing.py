from dataclasses import dataclass
from scipy.stats import nbinom, norm
import numpy as np
from metaCNV.metaCNV import _get_design_matrix
from metaCNV._cnv_model import _split_features
from metaCNV._entropy_model import get_model as _get_entropy_model


summarize_cols = ['coverage','n_reads','entropy','mutation_rate']

def _get_interval_features(interval, mask_fn = lambda x: x):

    interval = interval.copy()

    use_mask = mask_fn(interval)
    for col in summarize_cols:
        interval[col] = np.array(interval[col])[use_mask]
    
    X = _get_design_matrix(interval)
    cnvdata = _split_features(X, X.design_info)

    return cnvdata


def _get_continuity_groups(
    regions
):

    @dataclass
    class Region:
        contig : str; start : int; end : int; ploidy : float        

    def is_contiguous(region1, region2):
        return (
            region1.contig == region2.contig and
            region1.ploidy == region2.ploidy and
            region1.end == region2.start
        )
    
    continuity_groups = []
    group_id = 0
    current_region = None

    
    for _, row in regions.iterrows():
        next_region = Region(contig = row.contig, start = row.start, end = row.end, ploidy = row.ploidy)

        if current_region is None or is_contiguous(current_region, next_region):
            continuity_groups.append(group_id)
        else:
            group_id += 1
            continuity_groups.append(group_id)

        current_region = next_region

    return continuity_groups


def _aggregate_groups(regions):

    return regions.groupby('continuity_group').aggregate({
        'contig' : lambda x: x.iloc[0],
        'start' : min,
        'end' : max,
        'ploidy' : lambda x: x.iloc[0],
        **{col : list for col in summarize_cols}
    })


def _create_coverage_hypothesis_test(
        cnvdata,*,
        alternative_ploidy,
        n_samples = 10000, 
        max_fdr = 0.05,
        beta, 
        theta,
    ):

    alternative_ploidy = max(alternative_ploidy, 0.01)

    _lambda = np.exp(cnvdata.features @ beta.T)
    _ploidy_ratio = np.log(_lambda + theta) - np.log(alternative_ploidy*_lambda + theta)
    
    def lrt(y, prior_odds = 0):
        return (np.log(alternative_ploidy) + _ploidy_ratio) @ y + theta*_ploidy_ratio.sum() + prior_odds

    def convert_params(mu, theta):
        r = theta
        var = mu + 1 / r * mu ** 2
        p = (var - mu) / var
        return r, 1 - p

    y_tild = nbinom(*convert_params(_lambda, theta))\
        .rvs(size = (n_samples, _lambda.size)).T

    lrt_suffstat = lrt(y_tild)

    null_distribution = norm( np.mean(lrt_suffstat), np.sqrt(np.var(lrt_suffstat)) )
    critical_value = null_distribution.ppf(1 - max_fdr)
    critical_value = max(critical_value, 0)

    test_fdr = 1 - null_distribution.cdf(critical_value)
    
    return lambda cnvdata : lrt(cnvdata.n_reads, prior_odds=-critical_value), test_fdr, critical_value



def _create_entropy_hypothesis_test(
        cnvdata,*,
        alternative_ploidy, 
        max_fdr = 0.05, 
        n_samples = 10000,
        df = 30, 
        n_strains = 1,
):
    
    entropy_kw = dict(
        coverage = cnvdata.coverage,
        mutation_rate = cnvdata.mutation_rate,
        df = df,
    )
    
    null_model = _get_entropy_model(ploidy = n_strains, **entropy_kw)
    alternative_model = _get_entropy_model(ploidy = alternative_ploidy*n_strains, **entropy_kw)

    def lrt(entropy, prior_odds = 0):
        return alternative_model.logpdf(entropy) - null_model.logpdf(entropy) + prior_odds
    
    entropy_tild = null_model.rvs(size = (n_samples, cnvdata.entropy.size)).T
    lrt_suffstat = lrt(entropy_tild)

    null_distribution = norm( np.mean(lrt_suffstat), np.sqrt(np.var(lrt_suffstat)) )
    critical_value = null_distribution.ppf(1 - max_fdr)
    critical_value = max(critical_value, 0)

    test_fdr = 1 - null_distribution.cdf(critical_value)

    return lambda cnvdata : lrt(cnvdata.entropy, prior_odds=-critical_value), test_fdr, critical_value



def _apply_fdr_test(*,
                    interval, 
                    mask_fn, 
                    test_fn,
                    max_fdr = 0.05,
                    n_samples = 10000,
                    **test_kw,
                    ):

    cnvdata = _get_interval_features(interval, mask_fn= mask_fn)
    
    lr_test, fdr, critical_value = test_fn(cnvdata, 
                    n_samples = n_samples,
                    max_fdr = max_fdr,
                    alternative_ploidy = interval.ploidy, 
                    **test_kw,  
                    )
    
    likelihood_ratio = lr_test(cnvdata)

    return likelihood_ratio, fdr, critical_value



def fdr_tests(regions, model, max_fdr = 0.05):

    regions['continuity_group'] = _get_continuity_groups(regions)
    intervals = _aggregate_groups(regions)

    def _disaggregate(x):
        return list(map(list, x))

    intervals['coverage_test_statistic'], intervals['coverage_fdr'], intervals['coverage_critical_value'] \
                = _disaggregate( 
                    intervals.apply(_apply_fdr_test, 
                                axis = 1, 
                                beta = model.beta_,
                                theta = model.theta_,
                                max_fdr = max_fdr,
                                mask_fn = lambda interval : [n_reads >= 0 for n_reads in interval['n_reads']],
                                test_fn = _create_coverage_hypothesis_test,
                    )
                )

    intervals['entropy_test_statistic'], intervals['entropy_fdr'], intervals['entropy_critical_value'] \
                = _disaggregate(
                    intervals.apply(_apply_fdr_test,
                                axis = 1,
                                df = model.entropy_df,
                                n_strains = model.n_strains,
                                max_fdr = max_fdr,
                                mask_fn = lambda interval : [entropy >= 0 for entropy in interval['entropy']],
                                test_fn = _create_entropy_hypothesis_test,
                    )
                )


    intervals['coverage_supports_CNV'] = intervals.coverage_test_statistic > 0
    intervals['entropy_supports_CNV'] = intervals.entropy_test_statistic > 0
    intervals['supports_CNV'] = intervals.coverage_supports_CNV & intervals.entropy_supports_CNV

    return intervals


def get_summary(intervals):
    pass
