
import subprocess
import sys
import tempfile
import os
from multiprocessing.dummy import Pool
from math import log
import re
import tqdm

_process_kw = dict(
        text=True, 
        bufsize=10000,
        universal_newlines=True,
        stderr=sys.stderr,
    )


def call_consensus(*,reference, bamfile, output, min_quality=255):
    
    subprocess.Popen([
            'samtools','consensus','-f', reference,
            '--mode','simple',
            '-aa',
            '--show-ins','no',
            '--show-del','yes',
            '--use-qual',
            '--min-MQ',str(min_quality),
            '--min-depth','3',
            '--call-frac','0.6',
            bamfile,
        ],
        stdout=output,
        **_process_kw,
    ).communicate()


def _read_alleles(fastas,*,chrom, start, end, n_jobs = 1):

    def faidx(fasta):
        try:
            return subprocess.check_output(
                f'samtools faidx -n0 {fasta} "{chrom}:{int(start) + 1}-{end}"',
                shell=True,
                **_process_kw,
            ).split('\n')[1]
        except subprocess.CalledProcessError:
            return 'N'*(int(end)-int(start))
        

    with Pool(n_jobs) as pool:
        for res in pool.imap_unordered(faidx, fastas):
            yield res


class NoSamplesError(ValueError):
    pass
    
def _get_theta_hat(alleles):
    
    def alleles_equal(a1, a2):

        assert len(a1) == len(a2)

        for n1, n2 in zip(a1, a2):
            if not n1 == 'N' and not n2 == 'N':
                if not n1 == n2:
                    return False

        return True


    allele_list = []
    n_samples = 0
    
    for sample_allele in alleles:
        
        if not sample_allele.count('N') == len(sample_allele):
            n_samples+=1
            
            if len(allele_list) == 0:
                allele_list.append(sample_allele)
            else:
                for alt_allele in allele_list:
                    if alleles_equal(sample_allele, alt_allele):
                        break
                else:
                    allele_list.append(sample_allele)
    
    if n_samples <= 1:
        raise NoSamplesError('No samples with non-N alleles')
    
    return len(allele_list)/log(n_samples), allele_list, n_samples


## export to command line
def get_mutation_rate(fastas, window_size = 200, n_jobs = 1, 
                      output = sys.stdout, region = None):

    reference = fastas[0] #pull the first off the list as an example
    index = reference + '.fai'
    if not os.path.exists(index):
        subprocess.run(['samtools', 'faidx', reference], **_process_kw)

    with tempfile.NamedTemporaryFile() as windows_file:
        
        if region is None:
            # 0. make genome file
            genome_process = subprocess.Popen(
                ['cut', '-f', '1,2', index], 
                stdout=subprocess.PIPE, 
                **_process_kw
            )

            # 1. make genomic windows
            subprocess.Popen(
                        ['bedtools', 'makewindows', '-w', str(window_size),'-g', '-'],
                        stdin=genome_process.stdout,
                        stdout=windows_file,
                        **_process_kw,
                    ).communicate()
            
        else:
            chrom,start,end = re.split('[:-]',region)

            assert (int(end) - int(start)) % window_size == 0, 'Region must be a multiple of window size'

            subprocess.Popen(
                        ['bedtools', 'makewindows', '-w', str(window_size), '-b', '-'],
                        stdout=windows_file,
                        stdin=subprocess.PIPE,
                        **_process_kw,
                    ).communicate(
                        input = '\t'.join([chrom,start,end])
                    )
            
        num_regions = sum(1 for _ in open(windows_file.name, 'r'))

        # 2. get alleles
        with open(windows_file.name ,'r') as windows:

            print('#contig','#start','#end','#mutation_rate','#alleles','#samples', 
                  sep = '\t', file = output)

            for window in tqdm.tqdm(windows, total = num_regions, desc = 'Progress', ncols = 100):

                chrom, start, end = window.strip().split('\t')

                try:
                    mutation_rate, alleles, n_samples =_get_theta_hat(
                        _read_alleles(fastas, 
                                    chrom = chrom, start = start, end = end, 
                                    n_jobs=n_jobs
                                    )
                        )
                except NoSamplesError:
                    mutation_rate, alleles, n_samples = 0, [], 0

                print(chrom, start, end, mutation_rate, ','.join(alleles), n_samples,
                      sep = '\t', file = output)

