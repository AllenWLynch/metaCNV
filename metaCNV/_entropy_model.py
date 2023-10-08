from scipy.special import gammaln
import numpy as np
from itertools import product
from collections import Counter
from scipy.stats import binom, norm, t
import os
from pandas import DataFrame, read_csv


def partition_alleles(N):
    def partition_helper(N, current_partition):
        if N == 0:
            partitions.append(current_partition[:])
            return
        if N < 0:
            return

        for i in range(1, N + 1):
            if not current_partition or i <= current_partition[-1]:
                current_partition.append(i)
                partition_helper(N - i, current_partition)
                current_partition.pop()

    partitions = []
    partition_helper(N, [])
    return partitions


def log_ESF_prob(partition, N, theta):

    assert sum(partition) == N
    j = np.arange(1, N + 1)
    a_n = np.array( [partition.count(_j) for _j in j] )
    
    return gammaln(N + 1) + gammaln(theta) - gammaln(N + theta) \
        + np.sum( a_n * np.log(theta) -  a_n * np.log(j) - gammaln(a_n + 1) )


def get_p_variant(theta, total_population_size = 1e12, len_allele = 100):

    population_alleles = theta * np.log(total_population_size)
    assert population_alleles >= 2
    p_variant = 1/(2 * len_allele) * np.log2(population_alleles)
    return p_variant


def generate_num_variants_distribution(parition, p_variant):

    is_variant = np.array( list( product(range(2), repeat = len(parition)) ) )
    num_observed_variants = is_variant @ np.array(parition)

    num_mutations = is_variant.sum(axis = 1)

    log_prob = num_mutations * np.log(p_variant) + (len(parition) - num_mutations) * np.log(1 - p_variant)

    deduplicated = Counter()
    for num_observed, log_p in zip(num_observed_variants, log_prob):
        deduplicated[num_observed] += np.exp(log_p)

    return deduplicated


def marginalize_allele_sampling(N, theta, len_allele = 100, total_population_size = 1e12, condition_num_alleles = None):

    p_variant = get_p_variant(theta, len_allele = len_allele, total_population_size = total_population_size)
    
    marginalized_prob_num_variants = Counter()

    esf_samples = {tuple(p) : log_ESF_prob(p, N, theta) for p in partition_alleles(N)}
    if not condition_num_alleles is None:
        valid_partitions_ = [partition_ for partition_ in esf_samples.keys() if len(partition_) == condition_num_alleles]
        total_log_prob = np.log( sum([ np.exp(esf_samples[partition_]) for partition_ in valid_partitions_]) )
        
        esf_samples = { partition_ : log_prob - total_log_prob for partition_, log_prob in esf_samples.items() if partition_ in valid_partitions_ }
        
    for partition_, logp_partition in esf_samples.items():
        
        joint_prob_dict = Counter({ n_variants : np.exp( np.log(prob) + logp_partition )
                    for n_variants, prob in generate_num_variants_distribution(partition_, p_variant).items()
                    })
        marginalized_prob_num_variants =  marginalized_prob_num_variants + joint_prob_dict

    n_variants = list(marginalized_prob_num_variants.keys())
    prob = list(marginalized_prob_num_variants.values())

    return n_variants, prob



def get_expected_entropy(
    n_variants,
    p_n_variants,*,
    N,
    coverage = 90,
    error_rate = 0.01,
    moment = 1,
):
    
    def binary_entropy(p):
        q = 1-p
        ent = ( -( p * np.log2(p) + q*np.log2(q) ) )**moment

        return np.where(~np.isfinite(ent), 0., ent)


    def get_nucleotide_sampling_probs(p, _lambda):
        q = 1 - p
        return p*(1-_lambda) + q*_lambda + 2*p*_lambda/3
        
    
    expected_entropy = 0
    n_observed_variants = np.arange(0,coverage+1).astype(int)    
    empirical_entropy = binary_entropy(n_observed_variants/coverage)
    #empirical_entropy = np.abs(n_observed_variants/coverage - 0.5)
    for V, p_V in zip(n_variants, p_n_variants):    

        # account for errors
        p_variant_observed = get_nucleotide_sampling_probs(V/N, error_rate)
        
        expected_entropy+=np.sum( p_V * binom(coverage, p_variant_observed).pmf(n_observed_variants) * empirical_entropy )

    return expected_entropy



def simulate_distribution(
        max_ploidy = 16,
        max_coverage = 300,
        len_allele = 200,
        error_rate = 0.001
    ):

    results =[]

    for N in range(1,max_ploidy+1):
        for theta in [2**i for i in range(-3,5)]:

            n_var, prob = marginalize_allele_sampling(N, theta, len_allele=len_allele, total_population_size = 1e12)
            
            for cov in range(2, max_coverage):
                
                kw = dict(coverage = cov, N = N, error_rate = error_rate)

                mu_entropy = get_expected_entropy(n_var, prob, **kw)
                var_entropy = get_expected_entropy(n_var, prob, moment = 2, **kw) - mu_entropy**2

                results.append(
                        (N, theta, cov, mu_entropy, var_entropy/len_allele)
                )

    return DataFrame(results, 
                     columns = ['N', 'theta', 'coverage', 'mu_entropy', 'var_entropy']
                    )



def load_simulation_table():

    simulation_table_file = os.path.join(os.path.dirname(__file__), 'simulation_table.csv.gz')

    if not os.path.isfile(simulation_table_file):
        simulation_table = simulate_distribution()
        
        simulation_table.to_csv(simulation_table_file, 
                                index = False,
                                compression='gzip'
                                )
        
        return simulation_table
    
    else:
        return read_csv(simulation_table_file)\
            .set_index(['N','coverage','theta'])



SIMULATION_TABLE = load_simulation_table()

def get_model(
        ploidy,
        coverage,
        mutation_rate,
        df = 30,
    ):

    absolute_ploidy = ploidy.astype(int)
    coverage = np.rint(coverage).astype(int)
    mutation_rate = ( 2**np.clip( np.rint(np.log2(mutation_rate)), -2, 4 ) )

    combined_idx = list(zip(*[absolute_ploidy, coverage, mutation_rate]))

    return t(df,
            loc = SIMULATION_TABLE.loc[combined_idx].mu_entropy.values,
            scale = np.sqrt(SIMULATION_TABLE.loc[combined_idx].var_entropy.values)
            )


def model_log_likelihood(*,
        ploidy,
        entropy,
        coverage,
        mutation_rate,
        df = 30,
    ):

    return get_model(ploidy, coverage, mutation_rate, df = df).logpdf(entropy)