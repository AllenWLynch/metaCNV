
from _load_data import load_regions_df
import os

def main(metaCNV_file,
        ptr_prefix,
        strand_bias_tol = 0.1,
        min_entropy_positions = 0.9,
        ):
    
    with open(os.path.join(ptr_prefix, 'contigs.txt')) as f:
        contigs = f.read().splitlines()
    
    
    regions = load_regions_df(
                    metaCNV_file, contigs,
                    strand_bias_tol=strand_bias_tol,
                    min_entropy_positions=min_entropy_positions,
                )
         
    