Usage
-----

The workflow for using MetaCNV proceeds in three states: 

#. annotate a genome with mutation rates and origin information. 
#. extract features from a BAM file 
#. Call CNAs. 

First, set up a new environment and make sure you have samtools, bedtools, and tabix installed:

.. code-block:: bash

    $ conda create --name metaCNV -c conda-forge -c bioconda -y python=3.9 samtools bedtools tabix
    $ conda activate metaCNV

Then, install MetaCNV from github using pip:

.. code-block:: bash

    $ pip install git+https://github.com/AllenWLynch/metaCNV.git

Make sure to change the git URL above if you are using a fork.


Calculating origin distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial assumes you have a fasta file with a collection of microbial genomes, 
along with a BAM file containing reads mapped to those genomes.

First step, we must annotate some genome of interest with distances to the nearest origin of replication for each region. 
MetaCNV implements a simple algorithm to do this using GC-skew, but it requires that we use a whole genome reference.
In this case, I will call CNVs for the B. dorei full genome, which I have stored in "genomic.fa" (among other genomes). 
Run the "gc-skew" command:

.. code-block:: bash

    $ metaCNV gc-skew genomic.fa -o gc-skew.bedgraph.gz

This will produce a bedgraph file with the GC-skew for each region of the genome. Then:

.. code-block:: bash

    $ metaCNV ori-distance gc-skew.bedgraph.gz \
        -o ori-dist.bed \
        --contig GUT_GENOME143505_1 \
        --reference genomic.fa \

I split these operations into different commands because I suggest evaluating the skew for whole genome assemblies to
ensure that it behaves as expected. There may be inconsistencies if there was a mistake in the assembly,
in which case you may need to try a different strategy for calculating origin distances.

Calculating mutation rates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to calculate population-level mutation rates for each region of the genome, which requires a large database
of whole-genome multiple sequence alignments. One simple way of aggregating whole genome alignements is to call
consensus sequences from a collection of BAM files using the `samtools consensus` command. For a collection of MSAs,
use the `metaCNV mutation-rate` command:

.. code-block:: bash

    $ metaCNV mutation-rate \
        -o mutation-rates.bed \
        --n-jobs 5 \
        --region GUT_GENOME143505_1:1-1000000 \
        consesus/*.fa

This step can be quite time consuming. I suggest splitting the genome into smaller regions and running this command
in parallel on each, then merging the results. If you split the genome, make sure the intervals are divisible by the 
window size (default is 200).

Extracting features
~~~~~~~~~~~~~~~~~~~

With the annotations collected, we must next ingest a BAM file to prepare it for CNV calling. This command calculates 
and aggregates coverage and entropy statistics and filters out low-quality regions and reads:

.. code-block:: bash

    $ metaCNV extract-features \
        -o features.metaCNV
        --reference genomic.fa \
        --ori-distances ori-dist.bed \
        example.bam

There are a couple of things to keep in mind with this step:

* Ensure that all the chromosome names in the BAM file are present in the reference fasta. For example, use the fasta which the BAM file was aligned against. 
* Here, the ori distances are mapped to high-resolution bins for CNV calling. The resolution of the ORI distances may be higher (e.g. contig-level) depending on how that data was gathered.
* If you calculated mutation rates for a different window size, make sure to change the window size here as well.
  

Calling CNAs
~~~~~~~~~~~~

Finally, we can call CNAs using the features we extracted from the BAM file:

.. code-block:: bash

    $ metaCNV call \
        -o calls.bdorei \
        --mutation-rates mutation-rates.bed \
        --contigs GUT_GENOME143505_1 \
        features.metaCNV

Make sure to list all of the contigs associated with an organism in the `--contigs` argument! 
This commmand saves three files:

The `calls.bdorei.CNVs.bed` contains all candidate CNV calls in bed format, with additional columns:
* #coverage_test_statistic: the test statistic for the coverage statistical test, positive means the region supports a CNV
* #entropy_test_statistic: the test statistic for the entropy statistical test
* #coverage_supports_CNV: whether the coverage test supports a CNV
* #entropy_supports_CNV: whether the entropy test supports a CNV
* #support_CNV: whether both tests support a CNV (if copy number gain) or if coverage supports a CNV (if copy number loss)

The `calls.bdorei.regions.bed` file contains summary statistics for each region of the genome.

Finally, `calls.bdorei.summary.tsv` contains summary statistics and parameters for the model:

.. code-block:: bash

    $ cat calls.bdorei.summary.tsv

    log2_PTR        0.5016970404537802
    log_gc_effect   0.13548589835721667
    log_intercept_effect    3.6178352781643723
    dispersion      5.100218587147538
    num_candidate_CNVs      555.0
    num_supported_CNVs      471.0
    num_supported_CNVs_by_coverage  472.0
    num_supported_CNVs_by_entropy   110.0
    1_strain_AIC    129137.12887257658
    2_strain_AIC    135585.3194096952
    3_strain_AIC    139945.6237917519

Including the PTR effect estimate, the number of CNVs called, and the inferred number of strains. 

From these results, the code to produce the CNV plot figure in the manuscript is:

.. code-block:: python

    import pandas as pd
    import seaborn as sns
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    warnings.simplefilter('ignore')
    sns.set_style("whitegrid")

    regions = pd.read_csv('calls.bdorei.regions.bed', sep  ='\t')
    regions.columns = regions.columns.str.strip('#')

    intervals = pd.read_csv('calls.bdorei.CNVs.bed', sep = '\t')
    intervals.columns = intervals.columns.str.strip('#')

    fig, ax = plt.subplots(3,1,figsize=(15,5), sharex=True)
    plt.rcParams.update({'font.size': 14})

    j = 9000

    sns.scatterplot(
        y = regions.n_reads[:j],
        x = regions.start[:j],
        s = 3,
        alpha = 0.25,
        color = 'black',
        ax = ax[0],
        label = 'Coverage',
        legend=False,
    )


    for _, interval in intervals.iterrows():
        
        color = 'lightgrey'
        
        if interval.supports_CNV:
            color = 'black'
        
        if interval.start < j*200:
            sns.lineplot(
                y = [interval.ploidy, interval.ploidy],
                x = [interval.start, interval.end],
                ax = ax[1],
                color = color
            )


    ax[1].set_yticks([0,1,2,3])

    regions['log_mut'] = np.log2(regions.mutation_rate)

    sns.scatterplot(
        y = regions.entropy[:j],
        x = regions.start[:j],
        ax = ax[2],
        s = 2,
        hue = np.nan_to_num( np.log(regions.mutation_rate[:j]), -1.),
        palette='Blues',
        legend = False,
        hue_norm=(0,2),
    )
    ax[1].set(ylabel = 'Predicted\nploidy', ylim = (-0.1,3.1))
    ax[2].set(ylim = (0,0.1), ylabel = 'Entropy', xlabel = 'Position\n(dashed lines mark 25 Kb intervals)')
    ax[0].set(ylim = (1, 150), ylabel  = 'Coverage', xlim = (0, 200*j),
            xticks = range(0, 200*j, 25000), yticks = [30, 60, 90, 120])
    ax[2].set_xticklabels([])
    sns.despine(trim = True)
    ax[0].xaxis.grid(True, linestyle = '--', color = 'lightgrey', linewidth = 0.5)
    ax[1].xaxis.grid(True, linestyle = '--', color = 'lightgrey', linewidth = 0.5)
    ax[2].xaxis.grid(True, linestyle = '--', color = 'lightgrey', linewidth = 0.5)
    ax[0].yaxis.grid(False)
    ax[1].yaxis.grid(False)
    ax[2].yaxis.grid(False)

    ax[1].legend(
        [plt.Line2D([0], [0], color='black', lw=2, label='CNV'),
        plt.Line2D([0], [0], color='lightgrey', lw=2, label='No CNV')],
        ['Called CNV', 'Failed FDR test'],
        loc='upper right',
        frameon=False,
        fontsize = 12,
        bbox_to_anchor=(1.13, 1.05),
    )

    plt.colorbar(
        plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=2)),
        ax = ax[2],
        label = 'log mutation rate',
        orientation = 'horizontal',
        pad = 0.1,
        aspect = 10,
        ticks = [0,1,2],
        shrink = 0.5,
        fraction = 0.1,
        anchor = (0.75, 0.5),
    )


This code could be improved and added as a method to the CLI so that it is 
easy to generate these plots.