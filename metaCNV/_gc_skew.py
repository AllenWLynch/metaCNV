import subprocess
import tempfile
import sys
import os

_process_kw = dict(
        text=True, 
        bufsize=10000,
        universal_newlines=True,
        stderr=sys.stderr,
    )


# export to command line interface
def main(*,reference, output):

    index = reference + '.fai'
    if not os.path.exists(index):
        subprocess.run(['samtools', 'faidx', reference], **_process_kw)

    with tempfile.NamedTemporaryFile() as genome_file, \
        tempfile.NamedTemporaryFile() as window_gc, \
        open(output,'wb') as output:

        # 0. make genome file
        subprocess.run(['cut', '-f', '1,2', index], stdout=genome_file, **_process_kw)

        windows_process = subprocess.Popen(
            ['bedtools','makewindows','-w','100', '-g', genome_file.name],
            stdout=subprocess.PIPE,
            **_process_kw,
        )

        subprocess.Popen(
                ['bedtools','nuc','-fi',reference,'-bed','stdin'],
                stdin=windows_process.stdout,
                stdout=window_gc,
                **_process_kw,
            ).communicate()
        

        smoothed_windows_process = subprocess.Popen(
            ['bedtools','makewindows','-w','100000', 
            '-s', '100','-g', genome_file.name],
            stdout=subprocess.PIPE,
            **_process_kw,
        )

        average_gc_process = subprocess.Popen(
            ['bedtools','map','-a','stdin','-b',window_gc.name,'-c','7,8','-o','sum,sum'],
            stdin=smoothed_windows_process.stdout,
            stdout=subprocess.PIPE,
            **_process_kw,
        )

        skew_process = subprocess.Popen(
            ['awk','-v','OFS=\t','-v','windowsize=100',
            '{skew=($5-$4)/($5+$4); cumsum[$1]+=skew; print $1,$2+windowsize/2-50,$2+windowsize/2+50,skew,cumsum[$1]}'
            ],
            stdin=average_gc_process.stdout,
            stdout=subprocess.PIPE,
        )

        bgzip_process = subprocess.Popen(
            ['bgzip'],
            stdin=skew_process.stdout,
            stdout=output,
            **_process_kw,
        )

        bgzip_process.communicate()


def get_ori_distance(*,gc_skew_file, reference, contig, window_size, output):

    def middle(start, end):
        return int(start) + (int(end)-int(start))//2
    
    def shortest_distance_on_circle(point1, point2):

        # Calculate the distance in both directions (clockwise and counterclockwise)
        distance_clockwise = abs(point2 - point1)
        distance_counterclockwise = 1 - distance_clockwise

        # Choose the minimum of the two distances
        shortest_distance = min(distance_clockwise, distance_counterclockwise)

        return shortest_distance*2
    

    if not os.path.exists(gc_skew_file + '.tbi'):
        subprocess.run(['tabix', '-p', 'bed', gc_skew_file], **_process_kw)
    
    unzip = subprocess.Popen(
        ['tabix', gc_skew_file, f'{contig}:0-100000000'],
        stdout=subprocess.PIPE,
        **_process_kw,
    )
    
    origin = subprocess.check_output(
        ['awk','-v','OFS=\t','(!min[$1] || $5<=min[$1]){min[$1]=$5; minrow[$1]=$0} END{for (chrom in minrow) print minrow[chrom],"origin"}'],
        stdin=unzip.stdout,
        **_process_kw,
    )

    _, ori_start, ori_end = origin.strip().split('\t')[:3]

    length = None
    with open(reference + '.fai','r') as f:
        
        for line in f:
            chrom, length, *_ = line.strip().split('\t')
            if chrom == contig:
                length = int(length)
                break
        else:
            raise ValueError(f"Contig {contig} not found in reference {reference}")
    origin = middle(ori_start, ori_end)/length

    windows = subprocess.Popen(
        ['bedtools', 'makewindows','-w', str(window_size), '-g', '-'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        **_process_kw,
    )

    stdout, _ = windows.communicate(input=f'{contig}\t{length}\n')


    for line in stdout.splitlines():
        _, start, end = line.strip().split('\t')
        region = middle(start, end)/length

        print(
            contig, start, end,
            shortest_distance_on_circle(origin, region),
            sep='\t',
            file=output,
        )


        



    



