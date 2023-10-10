import subprocess
import sys
from math import log2
from collections import Counter
import re
import tempfile
import os


def _get_pileup_stats(chrom, pos, ref_base, num_reads, bases, baq):

    ref_base = ref_base.upper()
    num_reads = int(num_reads)

    if num_reads==0:
        return  chrom, int(pos) - 1, pos, 0., 0, 0, ':'.join(['0']*5)
    
    bases = re.sub('[\+\-]\d+[ATCGN]+', '', bases) #remove indels from base string

    check_groups = 'ATGCN*'
    base_string = bases.upper().replace(',',ref_base).replace('.', ref_base)
    basecalls = Counter( base_string )

    p_ref = basecalls[ref_base]/num_reads

    entropy = \
        -( p_ref* log2(p_ref) if p_ref > 0 else 0) \
        -( (1 - p_ref) * log2(1 - p_ref) if p_ref < 1 else 0)
        

    basecall_summary = ':'.join(map(str, [basecalls[base] for base in check_groups]))
    forward_strand = len([base for base in bases if base in 'ACGTN.'])

    return  chrom, int(pos) - 1, pos, entropy, num_reads, forward_strand, basecall_summary


## export to command line interface
def entropy(*,pileup, output = sys.stdout):
    for line in pileup:
        try:
            print(
                *_get_pileup_stats(*line.strip().split('\t')),
                sep = '\t',
                file = output
            )
        except BrokenPipeError:
            sys.exit(0)
    

## export to command line interface
def metaCNV_data(*,
    bamfile : str,
    reference : str,
    ori_distances : str,
    output : bytes,
    window_size = 200,
    min_quality = 20,
    ):

    _process_kw = dict(
        text=True, 
        bufsize=10000,
        universal_newlines=True,
        stderr=sys.stderr,
    )

    index = reference + '.fai'
    if not os.path.exists(index):
        subprocess.run(['samtools', 'faidx', reference], **_process_kw)

    with tempfile.NamedTemporaryFile() as windows, \
        tempfile.NamedTemporaryFile() as genome_file, \
        open(output,'wb') as output:

        # 0. make genome file
        subprocess.run(['cut', '-f', '1,2', index], stdout=genome_file, **_process_kw)

        # 1. make genomic windows
        windows_process = subprocess.Popen(
                        ['bedtools', 'makewindows', '-w', str(window_size),'-g', genome_file.name],
                        stdout=subprocess.PIPE,
                        **_process_kw,
                    )
        
        gc_process = subprocess.Popen(
            f'bedtools nuc -fi {reference} -bed - | cut -f 1-3,5 | tail -n +2',
            shell=True,
            stdin=windows_process.stdout,
            stdout=subprocess.PIPE,
            **_process_kw,
        )

        subprocess.Popen(
            ['bedtools','map','-sorted','-a','-','-b', ori_distances,
             '-c', '4', '-o', 'mean', '-null', '-1', '-g', genome_file.name],
            stdin=gc_process.stdout,
            stdout=windows,
            **_process_kw,
        ).communicate()

        # 3. Process pileup
        pileup_process = subprocess.Popen([
            'samtools', 'mpileup', '--reference', reference, 
            '--no-BAQ', '--min-MQ', str(min_quality), bamfile],
            stdout=subprocess.PIPE,
            **_process_kw
            )

        entropy_process = subprocess.Popen(
                ['metaCNV', 'pileup-entropy','-'],
                stdin=pileup_process.stdout,
                stdout= subprocess.PIPE,
                **_process_kw
                )

        aggregate_process = subprocess.Popen(
            ['bedtools', 'map', '-sorted', '-null', '0.', 
             '-g', genome_file.name,
             '-a', windows.name, '-b', 'stdin', 
             '-c', '4,5,6,5', '-o', 'collapse,collapse,sum,count'],
            stdin= entropy_process.stdout,
            stdout= subprocess.PIPE,
            **_process_kw,
        )
                
        gzip_process = subprocess.Popen(
            ['bgzip'],
            stdin=aggregate_process.stdout,
            stdout=output,
            **_process_kw,
        )

        gzip_process.communicate()

    subprocess.call(['tabix', '-s', '1', '-b', '2', '-e', '3','-0', output.name], **_process_kw)
