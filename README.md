# **ONTraC**

[![PyPI Latest Release](https://img.shields.io/pypi/v/ONTraC.svg)](https://pypi.org/project/ONTraC/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ONTraC.svg)](https://pypi.org/project/ONTraC/)
[![Downloads](https://static.pepy.tech/badge/ONTraC)](https://pepy.tech/project/ONTraC)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ONTraC.svg?label=PyPI%20downloads)](https://pypistats.org/packages/ontrac)
[![GitHub Stars](https://badgen.net/github/stars/gyuanlab/ONTraC)](https://github.com/gyuanlab/ONTraC)
[![GitHub Issues](https://img.shields.io/github/issues/gyuanlab/ONTraC.svg)](https://github.com/gyuanlab/ONTraC/issues)
[![GitHub License](https://img.shields.io/github/license/gyuanlab/ONTraC.svg)](https://github.com/gyuanlab/ONTraC/blob/master/LICENSE)

ONTraC (Ordered Niche Trajectory Construction) is a niche-centered, machine learning
method for constructing spatially continuous trajectories. ONTraC differs from existing tools in
that it treats a niche, rather than an individual cell, as the basic unit for spatial trajectory
analysis. In this context, we define niche as a multicellular, spatially localized region where
different cell types may coexist and interact with each other.  ONTraC seamlessly integrates
cell-type composition and spatial information by using the graph neural network modeling
framework. Its output, which is called the niche trajectory, can be viewed as a one dimensional
representation of the tissue microenvironment continuum. By disentangling cell-level and niche-
level properties, niche trajectory analysis provides a coherent framework to study coordinated
responses from all the cells in association with continuous tissue microenvironment variations.

![logo](docs/source/_static/images/logo_with_text_long.png)
![ONTraC Structure](docs/source/_static/images/ONTraC_structure.png)

## Installation

```sh
pip install ONTraC
```

For details and alternative approches, please see the [installation tutorial](https://ontrac-website.readthedocs.io/en/latest/installation.html)

## Tutorial

Please see [ONTraC website](https://ontrac-website.readthedocs.io/en/latest/) for details.

### Input File

A example input file is provided in `examples/stereo_seq_brain/meta_data.csv`.
This file contains all input formation with five columns: Cell_ID, Sample, Cell_Type, x, and y.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

For detailed information about input and output file, please see the [IO files explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/IO_files.html).

### Running ONTraC

The required options for running ONTraC are the paths to the input file and the three output directories:

- **NN-dir:** This directory stores preprocessed data and other intermediary datasets for analysis.
- **GNN-dir:** This directory stores output from he GNN algorithm.
- **NT-dir:** This directory stores NT output.

For detailed description about all parameters, please see the [Parameters explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/parameters.html).

```{sh}
ONTraC --meta-input simulated_dataset.csv --NN-dir simulation_NN --GNN-dir simulation_GNN --NT-dir simulation_NT --hidden-feats 4 -k 6 --modularity-loss-weight 0.3 --purity-loss-weight 300 --regularization-loss-weight 0.1 --beta 0.03 2>&1 | tee simulation.log
```

The input dataset and output files could be downloaded from the [Zenodo Dataset Repository](https://doi.org/10.5281/zenodo.11186619).

We recommand running `ONTraC` on GPU, it may take much more time on your own laptop with CPU only.

### Output

The intermediate and final results are located in `NN-dir`, `GNN-dir`, and `NT-dir` directories. Please see [IO files explanation](https://ontrac-website.readthedocs.io/en/latest/tutorials/IO_files.html#output-files) for detailed infromation.

### Visualization

Please see the [Visualization Tutorial](https://ontrac-website.readthedocs.io/en/latest/tutorials/visualization.html).

### Interoperability

ONTraC has been incorporated with [Giotto Suite](https://drieslab.github.io/Giotto_website/articles/ontrac.html).

## Citation

**Wang, W.\*, Zheng, S.\*, Shin, C. S., Chávez-Fuentes J. C.  & [Yuan, G. C.](https://labs.icahn.mssm.edu/yuanlab/)$**. [ONTraC characterizes spatially continuous variations of tissue microenvironment through niche trajectory analysis](https://doi.org/10.1186/s13059-025-03588-5). *Genome Biol*, 2025.

## Other Resources

- [Reproducible codes for our paper](https://github.com/gyuanlab/ONTraC_paper)
- [Dataset/output used in our paper](https://doi.org/10.5281/zenodo.11186619)
