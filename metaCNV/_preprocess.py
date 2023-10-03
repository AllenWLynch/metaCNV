import subprocess
import sys
from math import log2
from collections import Counter
import re
import tempfile
import os


_process_kw = dict(
    text=True, 
    bufsize=1000,
    universal_newlines=True,
)


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
    output : bytes,
    window_size = 200,
    min_quality = 20,
    subcontig_size = 100000,
    ):

    #genome_file = '/Users/allenwlynch/projects/metagenomes/mutation_analysis/realdata-test-1/genomic.chromsizes' #reference + '.fai'
    index = reference + '.fai'
    if not os.path.exists(index):
        subprocess.run(['samtools', 'faidx', reference], **_process_kw)

    with tempfile.NamedTemporaryFile() as windows, \
        tempfile.NamedTemporaryFile() as subcontigs_file, \
        tempfile.NamedTemporaryFile() as genome_file, \
        open(output,'wb') as output:

        # 0. make genome file
        subprocess.run(['cut', '-f', '1,2', index], stdout=genome_file, **_process_kw)

        # 1. make genomic windows
        windows_process = subprocess.Popen(
                        ['bedtools', 'makewindows', '-w', str(window_size),'-g', genome_file.name],
                        stdout=subprocess.PIPE,
                        stderr=sys.stderr,
                        **_process_kw,
                    )
        
        subprocess.Popen(
            f'bedtools nuc -fi {reference} -bed - | cut -f 1-3,5 | tail -n +2',
            shell=True,
            stdin=windows_process.stdout,
            stdout=windows,
            stderr=sys.stderr,
            **_process_kw,
        ).communicate()

        # 1. make low-resolution "subcontigs"
        subprocess.Popen(
                        ['bedtools', 'makewindows', '-w', str(subcontig_size),'-g', genome_file.name],
                        stdout=subcontigs_file,
                        stderr=sys.stderr,
                        **_process_kw,
                    ).communicate()

        # 3. Process pileup
        pileup_process = subprocess.Popen([
            'samtools', 'mpileup', '--reference', reference, 
            '--no-BAQ', '--min-MQ', str(min_quality), bamfile],
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            **_process_kw
            )

        scriptpath = os.path.join( os.path.dirname(__file__), 'cli.py' )
        entropy_process = subprocess.Popen(
                ['python3', scriptpath, 'entropy','-'],
                stdin=pileup_process.stdout,
                stdout= subprocess.PIPE,
                stderr=sys.stderr,
                **_process_kw
                )

        aggregate_process = subprocess.Popen(
            ['bedtools', 'map', '-sorted', '-null', '0.', 
             '-g', genome_file.name,
             '-a', windows.name, '-b', 'stdin', 
             '-c', '4,5,6,5', '-o', 'collapse,collapse,sum,count'],
            stdin= entropy_process.stdout,
            stdout= subprocess.PIPE,
            stderr=sys.stderr,
            **_process_kw,
        )
        
        intersect_process = subprocess.Popen(
            ['bedtools','intersect','-sorted', 
            '-f', str(0.5),
             '-wa', '-wb', '-g', genome_file.name,
             '-a', '-', '-b', subcontigs_file.name,
            ],
            stdin=aggregate_process.stdout,
            stdout= subprocess.PIPE,
            stderr=sys.stderr,
            **_process_kw,
        )
                
        gzip_process = subprocess.Popen(
            ['bgzip'],
            stdin=intersect_process.stdout,
            stdout=output,
            stderr=sys.stderr,
            **_process_kw,
        )

        gzip_process.communicate()

    subprocess.call(['tabix', '-s', '1', '-b', '2', '-e', '3','-0', output.name], **_process_kw)
