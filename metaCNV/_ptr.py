
import pandas as pd
from _robust_nmf import L21NMF_intercept
import numpy as np
from sklearn.linear_model import RANSACRegressor
from _load_data import load_regions_df
import os

class InsufficientCoverageError(Exception):
    pass


def _make_ptr_matrix_row(regions, 
                    max_median_coverage_deviation = 8, 
                    ):

    # 1. calculate the median coverage a cross regions and 
    median_coverage = regions['read_depth'].median()

    if not median_coverage > 1:
        raise InsufficientCoverageError(f"median_coverage = {median_coverage} is not greater than max_median_coverage_deviation = {max_median_coverage_deviation}")

    # 3. calculate the absolute log fold change in coverage
    regions['abs_lfc_median'] = np.abs( np.log1p(regions._read_depth/median_coverage) )

    # 4. filter out regions with a large deviation from the median coverage
    QC_pass_regions = regions[ regions.abs_lfc_median < max_median_coverage_deviation ]

    # 5. Aggregate QC pass regions by subcontig (essentially 25kb buckits), 
    #    and calculate the normalized coverage within the window
    #    e.g. the average of coverage for windows without extreme values
    groups = QC_pass_regions.groupby(['sample','subcontig'])
    subcontig_coverages = groups['read_depth'].sum()/groups['read_depth'].count()

    # 6. pivot the table so that each row is a sample, and each column is a subcontig
    subcontig_coverages.to_frame().reset_index().pivot(
        columns=['subcontig'],
        index='sample'
    )
    
    return subcontig_coverages


def _load_PTR_matrix(samples, contigs, 
                    strand_bias_tol = 0.1,\
                    max_median_coverage_deviation = 8,
                    ):

    ptr_matrix = []
    for sample in samples:

        try:
            new_row = _make_ptr_matrix_row(
                load_regions_df(
                    sample, contigs,
                    strand_bias_tol=strand_bias_tol,
                    min_entropy_positions=0.9,
                ),
                max_median_coverage_deviation=max_median_coverage_deviation,
            )
        except InsufficientCoverageError:
            pass
        else:
            ptr_matrix.append(new_row)

    if len(ptr_matrix) == 0:
        raise InsufficientCoverageError("No samples with sufficient coverage")
    
    ptr_matrix = pd.concat(ptr_matrix, axis=1).fillna(0.)
    return ptr_matrix


def get_mask(ptr_matrix):
    mask = ptr_matrix > 0
    return mask


def _get_genome_length_adjustment(X_oris):

    X = np.arange(len(X_oris))[:,np.newaxis]
    y = np.sort(X_oris)
    
    inliers = RANSACRegressor(residual_threshold=0.01).fit(X, y).inlier_mask_
    normal_bounds = (y[inliers].min(), y[inliers].max())

    genome_length_adjustment =  normal_bounds[1] - normal_bounds[0]

    return normal_bounds, genome_length_adjustment



def _infer_positional_ptr(ptr_matrix, mask = None):

    W, H = L21NMF_intercept(
            n_components=2, 
            max_iter=100000, 
            init='random',
            skip_iter=1000
        ).fit(np.log(ptr_matrix.values))
    
    X_ori_tentative = H[0,:]
    ori_bounds, genome_length_adjustment = _get_genome_length_adjustment(X_ori_tentative)

    PTR, abundance = W[:,0]*genome_length_adjustment, W[:,1]

    X_ori =  (X_ori_tentative - ori_bounds[0])/genome_length_adjustment
    X_ori = np.where(np.logical_or(X_ori < 0, X_ori>1), np.nan, X_ori)

    return {
        'PTR' : pd.Series(PTR, index = ptr_matrix.index, name='PTR'),
        'abundance' : pd.Series(abundance, index = ptr_matrix.index, name='abundance'),
        'X_ori' : pd.Series(X_ori, index = ptr_matrix.columns, name='X_ori')
    }



# export to command line
def learn_PTR_parameters(samples, contigs, 
                    strand_bias_tol = 0.1,\
                    max_median_coverage_deviation = 8,
                    outprefix = 'metaCNV'):
    
    ptr_matrix = _load_PTR_matrix(samples, contigs,
                    strand_bias_tol = strand_bias_tol,
                    max_median_coverage_deviation = max_median_coverage_deviation
                    )
    
    mask = get_mask(ptr_matrix)
    
    ptr_results = _infer_positional_ptr(ptr_matrix, mask = mask)

    if not os.path.isdir(outprefix):
        os.mkdir(outprefix)


    ptr_matrix.to_csv(os.path.join(outprefix, 'coverage_matrix.tsv'), sep='\t')
    ptr_results['PTR'].to_csv(os.path.join(outprefix,'PTR.tsv'), sep='\t')
    ptr_results['abundance'].to_csv(os.path.join(outprefix,'abundance.tsv'), sep='\t')
    ptr_results['X_ori'].to_csv(os.path.join(outprefix,'infered_ori_distance.tsv'), sep='\t')

    with open(os.path.join(outprefix,'contigs.txt','w')) as f:
        print(*contigs, sep='\n', file=f)