import pandas as pd
import tabix
import os
from statistics import mean
from scipy.special import betainc, comb
from scipy.special import beta as Beta

DTYPES_DICT = {
        'contig' : 'category',
        'sample' : 'category',
        'subcontig' : 'category',
        'start' : 'int32',
        'end' : 'int32',
        'n_reads' : 'float16',
        'num_entropy_positions' : 'uint16',
        '_n_reads' : 'uint16',
        '_n_forward' : 'uint16',
        'entropy' : 'float64',
        'gc_percent' : 'float16',
    }


def _incomplete_beta_binom(x, n, alpha, beta, p):
    posterior = (alpha + x, n-x+beta)
    return comb(n,x) * 1/Beta(alpha, beta) * Beta(*posterior) * betainc(*posterior, p)


def _strand_skew_test(n_reads, n_forward, strand_bias_tol = 0.1):
    
    n_forward = int(n_forward); n_reads = int(n_reads)

    # p is the proportion of reads on the forward strand, which should be around 0.5
    # H0: p in [0.5 - tol, 0.5 + tol]
    # H1: p in H0^c
    # 1. Place a uniform prior over the space of p (p ~ Beta(1,1))
    # 2. Convolve the prior with the binomial likelihood, only on the interval p in [0, v]
    # 3. Partition the space of F(x; v) into 3 regions: v=0.5 - tol, v=0.5 + tol, v=1
    #    Where v=1 is p(x) under the beta-binomial model
    # 4. Compute p_H0 = F(x; v=0.5+tol) - F(x; v=0.5-tol)
    beta_prior = (1,1)
    part1 = _incomplete_beta_binom(n_forward, n_reads, *beta_prior, 0.5 - strand_bias_tol)
    part2 = _incomplete_beta_binom(n_forward, n_reads, *beta_prior, 0.5 + strand_bias_tol)
    part3 = _incomplete_beta_binom(n_forward, n_reads, *beta_prior, 1.)

    prior = 0.9
    
    p_H0 = (part2 - part1)*prior
    p_H1 = (part3 - p_H0)*(1-prior)

    #returns true if the data is more likely under H1 than H0
    return p_H1 > p_H0



def load_regions_df(filename, contigs,
                 strand_bias_tol = 0.1,
                 min_entropy_positions = 0.9,
                 ):
    
    null_marker = float('nan')

    record_cols = ['contig', 'start', 'end', 'gc_percent','entropies', 'read_depths',
                          'n_forward_strand', 'num_windows',
                          'subcontig_chrom','subcontig_start','subcontig_end', 'sample']
    

    def mask_record(record):

        features = {
            'sample' : record['sample'],
            'contig' : record['contig'],
            'start' : record['start'],
            'end' : record['end'],
            'gc_percent' : record['gc_percent'],
            'subcontig' : (record['subcontig_chrom'], record['subcontig_start'], record['subcontig_end'])
        }

        window_size = int(record['end']) - int(record['start'])
        read_depths = list( map(float, record['read_depths'].split(',')) )
        n_forward = float(record['n_forward_strand'])/100
        n_reads = sum(read_depths)/100

        # store some raw data for debugging
        features['_n_forward'] = n_forward
        features['_n_reads'] = n_reads

        # check to see if this region contains a highly skewed strand bias,
        # if so, the basecalls and coverage are unreliable
        # exlcude this region from analysis by setting coverage and entropy to NaN
        if _strand_skew_test( n_reads, n_forward, strand_bias_tol = strand_bias_tol):
            
            features['n_reads'] = null_marker
            features['entropy'] = null_marker
            features['num_entropy_positions'] = 0

        else:

            features['n_reads'] = n_reads # number of reads divided by approximate length of read
            #read_depth_dispersion  = var(read_depth)/mean(read_depth)

            entropies = list( map(float, record['entropies'].split(',')) )
            # need at least 2 reads covering a base to have any entropy
            entropies = [e for e,c in zip(entropies, read_depths) if c >= 2] 

            # if there are not enough entropy positions, set entropy to NaN
            if (len(entropies) < min_entropy_positions*window_size):
                features['entropy'] = null_marker
                features['num_entropy_positions'] = 0
            else:
                features['entropy'] = mean(entropies)
                features['num_entropy_positions'] = len(entropies)

        return features
    

    features = []
    tb = tabix.open(filename)

    for contig in contigs:
        for _, record in enumerate( tb.query(contig, 0, int(1e9)) ):
            features.append(
                mask_record( dict(zip(record_cols, [*record, os.path.basename(filename)])) )
            )

    df = pd.DataFrame(features)
    # optimize memory usage of dataframe
    df = df.astype(DTYPES_DICT)
    return df


'''def load_samples(samples, contigs,
                 strand_bias_tol = 0.1,
                 min_entropy_positions = 0.9
                 ):

    sample_features = []
    for sample in samples:
        sample_features.append(
            load_contigs(sample, contigs,
                         strand_bias_tol = strand_bias_tol,
                         min_entropy_positions = min_entropy_positions
                        )
        )

    return pd.concat(sample_features, ignore_index = True)'''