
from _load_data import load_regions_df
import _cnv_model as model
import os
import numpy as np
import pandas as pd

def main(*,
        metaCNV_file,
        ptr_prefix,
        mutation_rates, 
        outfile,
        strand_bias_tol = 0.1,
        min_entropy_positions = 0.9,
        max_strains = 3,
        transition_matrix = None,
        alpha = 1e5,
        ):
    
    sample_name = os.path.basename(metaCNV_file)

    with open(os.path.join(ptr_prefix, 'contigs.txt')) as f:
        contigs = f.read().splitlines()
    
    mutation_rates = pd.read_csv(mutation_rates, sep = '\t')

    try:
        log_ptr = pd.read_csv(
                os.path.join(ptr_prefix, 'PTR.tsv'), 
                sep = '\t'
            )['PTR'].to_dict()[sample_name]
    except KeyError:
        raise KeyError(f'PTR for sample {sample_name} not found in {ptr_prefix}')

    ori_distance = pd.read_csv(
                    os.path.join(ptr_prefix, 'infered_ori_distance.tsv'), 
                    sep = '\t'
                )

    regions = load_regions_df(
                    metaCNV_file, 
                    contigs,
                    strand_bias_tol=strand_bias_tol,
                    min_entropy_positions=min_entropy_positions,
                )
    
    regions = regions.set_index('subcontig').join(
                    ori_distance, how = 'left'
                ).reset_index()
    
    assert regions.X_ori.notnull().sum() > len(regions)/2, 'Too many subcontigs (>1/2) have no inferred distance to origin'
    
    regions['offset'] = regions.X_ori*log_ptr
    regions['n_reads'] = np.rint(regions.n_reads)

    window_size = int(regions.iloc[0].end - regions.iloc[0].start)
    regions['coverage'] = np.rint(regions.n_reads * 100/window_size)


    if transition_matrix is None:
        t = pd.read_csv(
            transition_matrix, sep ='\t',
        )
        transition_matrix = model.TransitionMatrix(ploidies=t.columns, mat=t.values)
    else:
        transition_matrix = model._sparse_transition_matrix(alpha = alpha)

    regions = regions.sort_values('contig')
    _, lengths = np.unique(regions.contig, return_counts=True)


    bics = []; models = []

    for n_strains in range(1, max_strains + 1):
        
        n_strains_transition_matrix = transition_matrix._make_transition_matrix(n_strains)
        start_prob = model._get_stationary_state(n_strains_transition_matrix.mat)

        model = model.HMMModel.get_hmm_model(
            regions,
            n_strains = n_strains,
            transmat=n_strains_transition_matrix.mat,
            ploidies=n_strains_transition_matrix.ploidies,
            startprob=start_prob,
            training=True
        )

        model.fit(regions, lengths = lengths)

        models.append(model)
        bics.append(model.bic(regions, lengths = lengths))


    best_model = models[np.argmin(bics)]
    n_strains = np.argmin(bics) + 1

    regions['ploidy'] = best_model.predict(regions, lengths = lengths)*n_strains

    regions[['contig', 'start', 'end', 'ploidy', 'n_reads','entropy','offset']]\
            .to_csv(outfile, sep = '\t')


    