from dataclasses import dataclass
from scipy.stats import nbinom, norm
import numpy as np
from metaCNV._cnv_model import _split_features, _get_design_matrix
from metaCNV._entropy_model import get_model as _get_entropy_model


summarize_cols = ['coverage','n_reads','entropy','mutation_rate','gc_percent', 'ori_distance']


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


def _get_critical_value(null_lrt_samples, max_fdr = 0.05):

    n_samples = len(null_lrt_samples)
    cdf_value = int( n_samples * (1-max_fdr) )

    null_lrt_samples = np.sort(null_lrt_samples)
     
    critical_value = null_lrt_samples[cdf_value]
    
    #if critical_value > 0:
    test_fdr = max_fdr
    #else:
    #    critical_value = 0
    #    test_fdr = (null_lrt_samples > 0).sum().item()/n_samples

    return critical_value, test_fdr


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

    critical_value, test_fdr = _get_critical_value(lrt_suffstat, max_fdr = max_fdr)
    
    return lambda cnvdata : lrt(cnvdata.n_reads, prior_odds=-critical_value).item(), critical_value, test_fdr



def _create_entropy_hypothesis_test(
        cnvdata,*,
        alternative_ploidy, 
        max_fdr = 0.05, 
        df = 30, 
        n_strains = 1,
):
    
    n_samples = int( 100/max_fdr )
    
    entropy_kw = dict(
        coverage = np.maximum(cnvdata.coverage,2.),
        mutation_rate = cnvdata.mutation_rate,
        df = df,
    )
    
    n_strains_vector = np.ones(len(cnvdata.entropy)) * n_strains

    null_model = _get_entropy_model(ploidy = n_strains_vector, **entropy_kw)
    alternative_model = _get_entropy_model(ploidy = alternative_ploidy*n_strains_vector, **entropy_kw)

    def lrt(entropy, prior_odds = 0):
        return alternative_model.logpdf(entropy).sum(axis = 0) \
                    - null_model.logpdf(entropy).sum(axis = 0) + prior_odds
    
    
    entropy_tild = null_model.rvs(size = (cnvdata.entropy.size, n_samples))
    lrt_suffstat = lrt(entropy_tild)

    critical_value, test_fdr = _get_critical_value(lrt_suffstat, max_fdr = max_fdr)    

    return lambda cnvdata : lrt(cnvdata.entropy[:,np.newaxis], prior_odds=-critical_value).item(), critical_value, test_fdr



def _get_interval_features(interval, mask_fn = lambda x: x):

    interval = interval.copy()

    use_mask = mask_fn(interval)
    for col in summarize_cols:
        interval[col] = np.array(interval[col])[use_mask]
    
    X = _get_design_matrix(interval)
    cnvdata = _split_features(X, X.design_info)

    return cnvdata



def _apply_fdr_test(interval,*,
                    mask_fn, 
                    test_fn,
                    max_fdr = 0.05,
                    **test_kw,
                    ):

    cnvdata = _get_interval_features(interval, mask_fn= mask_fn)

    if len(cnvdata) == 0:
        return np.nan, np.nan
    
    lr_test, _, test_fdr = test_fn(
                    cnvdata, 
                    max_fdr = max_fdr,
                    alternative_ploidy = interval.ploidy, 
                    **test_kw,  
                    )
    
    likelihood_ratio = lr_test(cnvdata)

    return likelihood_ratio, test_fdr



def fdr_tests(regions, model, max_fdr = 0.05):

    regions['continuity_group'] = _get_continuity_groups(regions)
    intervals = _aggregate_groups(regions)
    intervals = intervals[intervals.ploidy != 1]

    def _disaggregate(x):
        return list(map(list, x))

    intervals[['coverage_test_statistic', 'coverage_fdr']] \
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

    # entropy test only makes sense for ploidy > 1
    duplications = intervals[intervals.ploidy > 1].copy()
    duplications[['entropy_test_statistic', 'entropy_fdr']] \
                = _disaggregate(
                    duplications.apply(_apply_fdr_test,
                                axis = 1,
                                df = model.entropy_df,
                                n_strains = model.n_strains,
                                max_fdr = max_fdr,
                                mask_fn = lambda interval : [entropy >= 0 for entropy in interval['entropy']],
                                test_fn = _create_entropy_hypothesis_test,
                    )
                )

    intervals = intervals.join(duplications[['entropy_test_statistic','entropy_fdr']])

    intervals['coverage_supports_CNV'] = intervals.coverage_test_statistic > 0
    intervals['entropy_supports_CNV'] = intervals.entropy_test_statistic > 0
    intervals['supports_CNV'] = (intervals.ploidy > 1) & (intervals.coverage_supports_CNV & intervals.entropy_supports_CNV) \
                                    | (intervals.ploidy < 1) & intervals.coverage_supports_CNV

    return intervals
