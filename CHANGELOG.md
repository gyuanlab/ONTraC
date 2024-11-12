# Change log

## [1.1.0] - XXXX-XXX-XX

Added:

- Low resolution (spot-based) data support
  - running
  - analysis
- External modules (STDeconvolve)

Changed:

- Make options and modules' name consistent
  - NN/niche network
  - GNN/graph neural network
  - NT/niche trajectory
  - Rename `createDataSet` to `ONTraC_NN`
  - Rename `NicheTrajectory` to `ONTraC_NT`
  - Rename `ONTraC_GP` to `ONTraC_GT`
- Refine the optparser structures
  - merge analysis optparser to main optparser module
- Extract preprocessing modules

Fixed:

- `TSP` method for Niche Trajectory construction
- Asymmeteric niche cluster connectivities

## [1.0.7] - 2024-Oct-23

Added:

- Niche cluster connectivity visualization for each sample
- Node colorbar for cluster connectivity visualization
- `scale factor` controlling the size of spatial-based plots
- `N_GCN_LAYERS` controlling the number of GCN layers in the model

Changed:

- Make `log` file optional for `ONTraC_analysis`

Fixed:

- Issues caused `ONTraC_GNN` still need dataset input

## [1.0.6] - 2024-Sep-20

Added:

- `ONTraC_GNN` for running GNN only
- GNN parameters group validation
- More functions descriptions under GNN part
- Transparent edges and colorbar for niche cluster connectivities visualization
- `TSP` method for Niche Trajectory construction

Changed:

- rename `GP` to `ONTraC_GP`

Fixed:

- Loading dataset error in `NicheTrajectory`
- Log printing error on Windows

## [1.0.5] - 2024-Sep-8

Added:

- Niche trajectory visualization suppression parameter

## [1.0.4] - 2024-Sep-3

Added:

- Integrate mudole with other workflow

## [1.0.3] - 2024-Aug-25

Added:

- GitHub conda building workflow

## [1.0.2] - 2024-Aug-21

Added:

- test module
- GitHub pytest workflow

## [1.0.1] - 2024-July-30

Added:

- `citation` info

## [1.0.0] - 2024-July-10

Added:

- `N_LOCAL` parameter
- Parameters validation for `niche_net_constr`

Changed:

- Support Python 3.10 and 3.12
- Refined Package structures
- Change the name of `NTScore` to `NicheTrajectory`
- Skip violinplot about cell type density along NT score in case of more than 100 cell types input

## [0.0.9] - 2024-Jun-20

Changed:

- Make the default value of beta consistent with our paper
- Update tutorials

Fixed:

- Input table handling: support int format for sample

## [0.0.8] - 2024-Jun-10

Added:

- Cell type composition visualization suppression parameter
- Niche cluster loadings visualization suppression parameter

Changed:

- Instruction for `analysis` installation
- Make log file and train loss in `analysis` part optional
- Get `edge_index` directly in `niche_net_constr` module
- Make losses name consistent with the paper

Fixed:

- Incorrect version display

## [0.0.7] - 2024-May-13

Added:

- Citation information
- New parameter `sample` in `analysis` module for plotting by sample
- Multiple cell type check in input data
- More detailed log output

Changed:

- Flexible figure size for some visualization
- IO options validation and output directories creation logic
- Device validate and use logic
- dataset load logic

Fixed:

- Fixed cell type composition read in analysis
- SpatialOmicsDataset using old saved data

Removed:

- Moved example dataset to Zenodo

## [0.0.6] - 2024-May-5

Added:

- Added `ONTraC_analysis` and `analysis` module
- Added `niche cluster` part in `GP.py`
- Added version information at the top of the output

Changed:

- Update package structures
- Update imports in `createDataSet.py`, `GP.py`, and `NTScore.py`
- Update package buildup settings in `pyproject.toml` and `MANIFEST.in`

Fixed:

- Fixed bug in `createDataSet.py`

## [0.0.5] - 2024-Apr-25

Added:

- Added `installation` tutorial
- Added `reproducible codes` for paper

Changed:

- Refactored the `pyproject.toml` according to setuptools
- Rename losses name to make them consistent with paper

Removed:

- Removed Unused imports and files
- Removed `setup.py`

## [0.0.4] - 2024-Apr-16

Added:

- Added `simulation` data and tutorial
- Added `niche cluster` information output
- Added `niche cluster` tutorial
- Added duplicate `Cell_ID` handle

Changed:

- Make this repository public

Fixed:

- Fixed errors when there is only 1 sample for `post-analysis` tutorial

## [0.0.3] - 2024-Apr-2

Added

- Added `post-analysis` tutorial

Changed:

- Updated dependent packages information
- Updated installation tutorial
- Updated `stereo-seq` example

## [0.0.2] - 2024-Mar-12

Added

- Added package description
- Added `.gitignore` to remove unnecessary files
- Added `ONTraC` for running all steps together
- Added `NTScore` for generating NTScore from GNN output
- Added pip installation support

Changed:

- New environment constructions
- Running process
- Uniform parameters control
- Output directories
- `createDataSet` generate cell type composition and GNN input from raw input
- Input data format
