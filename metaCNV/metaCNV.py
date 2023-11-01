
from metaCNV._load_data import format_data as _format_data
import metaCNV._cnv_model as model
from metaCNV._cnv_model import _get_design_matrix
from metaCNV._fdr_testing import fdr_tests
import numpy as np
import pandas as pd
import sys
import warnings
warnings.simplefilter('ignore')

import logging
logging.basicConfig(
    level = logging.INFO, format = '%(name)s - %(levelname)s - %(message)s', 
    datefmt = '%m/%d/%Y %I:%M:%S %p'
)
logger = logging.getLogger('metaCNV')


def _load_transition_matrix(transition_matrix = None, alpha = 5e7):

    if not transition_matrix is None:
        t = pd.read_csv(
            transition_matrix, sep ='\t',
        )
        transition_matrix = model.TransitionMatrix(ploidies=t.columns, mat=t.values)
    else:
        logger.info('No transition matrix provided, using sparse penalized matrix.')
        transition_matrix = model._sparse_transition_matrix(alpha = alpha)

    return transition_matrix



def _select_model(*,
        regions,
        lengths,
        max_strains,
        transition_matrix, 
        use_coverage = True,
        use_entropy = True,
        coverage_weight = 0.75,    
        entropy_df = 30,
    ):

    X = _get_design_matrix(regions)

    scores = []; models = []

    for n_strains in range(1, max_strains + 1):
        
        logger.info(f'Trying {n_strains} strain model ...')
        n_strains_transition_matrix = model._make_transition_matrix(transition_matrix, n_strains)
        start_prob = model._get_stationary_state(n_strains_transition_matrix.mat)

        hmm = model.HMMModel.get_hmm_model(
            X,
            n_strains=n_strains,
            transmat=n_strains_transition_matrix.mat,
            ploidies=n_strains_transition_matrix.ploidies,
            startprob=start_prob,
            training=True,
            use_coverage=use_coverage,
            use_entropy=use_entropy,
            coverage_weight=coverage_weight,
            entropy_df=entropy_df,
            verbose = True,
        )

        print(' Iteration  log_prob      log_liklihood_diff', file = sys.stderr)
        hmm.fit(X, lengths = lengths)

        models.append(hmm)

        strains_log_prior = np.log(0.95) if n_strains == 1 else np.log(0.05/(max_strains-1))
        scores.append(hmm.bic(X, lengths = lengths) +  strains_log_prior)

    best_model = models[np.argmin(scores)]
    n_strains = np.argmin(scores) + 1
    logger.info(f'Selected {n_strains} strain model with AIC = {scores[n_strains-1]}')

    return best_model, scores


def _get_summary(intervals, model, scores):
    
    summary = {
        'log2_PTR' : np.log2( np.exp( np.abs(model.beta_[1]) ) ),
        'log_gc_effect' : model.beta_[2],
        'log_intercept_effect' : model.beta_[0],
        'dispersion' : model.theta_,
        'num_candidate_CNVs' : len(intervals),
        'num_supported_CNVs' : intervals.supports_CNV.sum(),
        'num_supported_CNVs_by_coverage' : intervals.coverage_supports_CNV.sum(),
        'num_supported_CNVs_by_entropy' : intervals.entropy_supports_CNV.sum(),
        **{
            f'{n_strains}_strain_AIC' : score for n_strains, score in enumerate(scores, 1)
        }
    }

    return pd.Series(summary)


def _write_results(outprefix,*,model, intervals, regions, scores):
    
    intervals_df = intervals[[
        'contig', 'start', 'end', 'ploidy', 
        'coverage_test_statistic', 'entropy_test_statistic',
        'coverage_supports_CNV', 'entropy_supports_CNV', 'supports_CNV'
    ]]

    intervals_df.columns = '#' + intervals_df.columns
    intervals_df.to_csv(
        f'{outprefix}.CNVs.bed', sep = '\t', index = False,
    )

    summary_df = _get_summary(intervals, model, scores)
    summary_df.to_csv(
        f'{outprefix}.summary.tsv', sep = '\t', header=None,
    )

    regions_df = regions[[
        'contig', 'start', 'end', 'ploidy', 'n_reads', 'entropy', 'predicted_n_reads',
        'gc_percent','ori_distance','mutation_rate',
    ]]
    regions_df.columns = '#' + regions_df.columns
    regions_df.to_csv(
        f'{outprefix}.regions.bed', sep = '\t', index = False,
    )



def call_CNVs(*,
        metaCNV_file : str,
        mutation_rates : str, 
        outprefix : str,
        contigs : list[str],
        # model parameters
        coverage_weight = 0.8, 
        entropy_df = 20,
        max_fdr = 0.05,
        use_coverage = True,
        use_entropy = True,
        # data parameters
        strand_bias_tol = 0.1,
        min_entropy_positions = 0.9,
        max_strains = 2,
        transition_matrix = None,
        alpha = 5e7,
        ):
    
    logger.info(f'Loading data ...')
    regions, lengths = _format_data(
        metaCNV_file = metaCNV_file,
        mutation_rates = mutation_rates, 
        contigs = contigs,
        strand_bias_tol = strand_bias_tol,
        min_entropy_positions = min_entropy_positions,
    )

    transition_matrix = _load_transition_matrix(
        transition_matrix = transition_matrix,
        alpha = alpha,
    )

    logger.info(f'Learning coverage parameters')
    best_model, scores = _select_model(
        regions = regions,
        lengths = lengths,
        transition_matrix = transition_matrix,
        max_strains = max_strains,
        use_coverage = use_coverage,
        use_entropy = use_entropy,
        coverage_weight = coverage_weight,
        entropy_df = entropy_df,
    )
    
    regions['ploidy'], regions['predicted_n_reads'] = best_model.predict(_get_design_matrix(regions), lengths)
    
    logger.info(f'Applying FDR correction ...')
    intervals = fdr_tests(regions, model = best_model, max_fdr = max_fdr)

    logger.info(f'Writing results ...')
    _write_results(outprefix,
                   model = best_model, 
                   intervals = intervals, 
                   regions = regions,
                   scores = scores,
                )

    logger.info(f'Done!')
    