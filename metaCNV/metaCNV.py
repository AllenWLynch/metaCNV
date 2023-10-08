
from metaCNV._load_data import format_data as _format_data
import metaCNV._cnv_model as model
import numpy as np
import pandas as pd
from patsy import dmatrix
from metaCNV._fdr_testing import fdr_tests, get_summary


import logging
logging.basicConfig(
    level = logging.INFO, format = '%(name)s - %(levelname)s - %(message)s', 
    datefmt = '%m/%d/%Y %I:%M:%S %p'
)
logger = logging.getLogger('metaCNV')


def _get_design_matrix(regions):

    return dmatrix(
                'coverage + n_reads + entropy + mutation_rate + ' \
                'I(-ori_distance + 0.5) + standardize(np.log(gc_percent)) + 1', 
                data = regions,
                )


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
            n_strains = n_strains,
            transmat=n_strains_transition_matrix.mat,
            ploidies=n_strains_transition_matrix.ploidies,
            startprob=start_prob,
            training=True,
            use_coverage=use_coverage,
            use_entropy=use_entropy,
            coverage_weight=coverage_weight,
            entropy_df = entropy_df,
        )

        hmm.fit(X, lengths = lengths)

        models.append(hmm)
        scores.append(hmm.bic(X, lengths = lengths))

    best_model = models[np.argmin(scores)]
    logger.info(f'Selected {n_strains} strain model with AIC = {scores[n_strains-1]}')

    return best_model, scores


def metaCNV(*,
        metaCNV_file : str,
        mutation_rates : str, 
        outfile : str,
        contigs : list[str],
        # model parameters
        coverage_weight = 0.75, 
        entropy_df = 30,
        max_fdr = 0.05,
        use_coverage = True,
        use_entropy = True,
        # data parameters
        strand_bias_tol = 0.1,
        min_entropy_positions = 0.9,
        max_strains = 3,
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
    
    regions['ploidy'], regions['predicted_coverage'] = best_model.predict(_get_design_matrix(regions), lengths)
    
    logger.info(f'Applying FDR correction ...')
    intervals = fdr_tests(regions, model = best_model, max_fdr = max_fdr)

    return {
        'model' : best_model, 
        'scores' : scores,
        'regions' : regions,
        'intervals' : intervals
    }


    