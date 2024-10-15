# Parameters explanation

## Full parameters list

### Full parameters for ONTraC

```{text}
Usage: ONTraC <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> <-d DATASET>
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR]
    [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT]
    [--purity-loss-weight PURITY_LOSS_WEIGHT] [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]
    [--trajectory-construct TRAJECTORY_CONSTRUCT]

All steps of ONTraC including dataset creation, Graph Pooling, and NT score
calculation.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.
    -d DATASET, --dataset=DATASET
                        Original input dataset.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset
                        preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction.
                        It should be less than the number of cells in each
                        sample. Default is 50.
    --n-local=N_LOCAL   Specifies the nth closest local neighbors used for
                        gaussian distance normalization. It should be less
                        than the number of cells in each sample. Default is
                        20.

  Options for GNN training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto
                        select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is
                        1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is
                        100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50.
                        Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole
                        dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    -k K, --k-clusters=K
                        Number of niche clusters. Default is 6.
    --modularity-loss-weight=MODULARITY_LOSS_WEIGHT
                        Weight for modularity loss. Default is 0.3.
    --purity-loss-weight=PURITY_LOSS_WEIGHT
                        Weight for purity loss. Default is 300.
    --regularization-loss-weight=REGULARIZATION_LOSS_WEIGHT
                        Weight for regularization loss. Default is 0.1.
    --beta=BETA         Beta value control niche cluster assignment matrix.
                        Default is 0.03.

  Options for niche trajectory:
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is
                        'BF' (brute-force). A faster alternative is 'TSP'.
```

### Full parameters for createDataSet

```{text}
Usage: createDataSet <--preprocessing-dir PREPROCESSING_DIR> <-d DATASET> [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL]

Create dataset for follwoing analysis.

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    -d DATASET, --dataset=DATASET
                        Original input dataset.

  Niche Network Construction:
    --n-cpu=N_CPU       Number of CPUs used for parallel computing in dataset
                        preprocessing. Default is 4.
    --n-neighbors=N_NEIGHBORS
                        Number of neighbors used for kNN graph construction.
                        It should be less than the number of cells in each sample.
                        Default is 50.
    --n-local=N_LOCAL
                        Specifies the nth closest local neighbors used for
                        gaussian distance normalization. It should be less than
                        the number of cells in each sample. Default is 20.
```

### Full parameters for ONTraC_GNN

```{text}
Usage: ONTraC_GNN <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]

Graph Neural Network (GNN)

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Options for training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto
                        select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is
                        1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is
                        100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50.
                        Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole
                        dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    -k K, --k-clusters=K
                        Number of niche clusters. Default is 6.
    --modularity-loss-weight=MODULARITY_LOSS_WEIGHT
                        Weight for modularity loss. Default is 0.3.
    --purity-loss-weight=PURITY_LOSS_WEIGHT
                        Weight for purity loss. Default is 300.
    --regularization-loss-weight=REGULARIZATION_LOSS_WEIGHT
                        Weight for regularization loss. Default is 0.1.
    --beta=BETA         Beta value control niche cluster assignment matrix.
                        Default is 0.03.
```

### Full parameters for NicheTrajectory

```{text}
Usage: NicheTrajectory <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> 
            [--trajectory-construct TRAJECTORY_CONSTRUCT]

Niche trajectory: construct niche trajectory for niche cluster and project the NT score to each cell

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.

  Options for niche trajectory:
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is
                        'BF' (brute-force). A faster alternative is 'TSP'.
```

### Full parameters for ONTraC_analysis

```{text}
Usage: ONTraC_analysis <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR>
    <--NTScore-dir NTSCORE_DIR> <-o OUTPUT_DIR> [-l LOG_FILE] [-r REVERSE] [-s SAMPLE] [--scale-factor SCALE_FACTOR]
    [--suppress-cell-type-composition] [--suppress-niche-cluster-loadings] [--suppress-niche-trajectory]

Analysis the results of ONTraC.

Options:
  --version             show program's version number and exit
  -h, --help            Show this help message and exit.
  -o OUTPUT, --output=OUTPUT
                        Output directory.
  -l LOG, --log=LOG     Log file.
  -r, --reverse         Reverse the NT score.

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.
    -d DATASET, --dataset=DATASET
                        Original input dataset.

  Visualization options:
    -s, --sample        Plot each sample separately.
    --scale-factor=SCALE_FACTOR
                        Scale factor control the size of spatial-based plots.

  Suppress options:
    --suppress-cell-type-composition
                        Suppress the cell type composition visualization.
    --suppress-niche-cluster-loadings
                        Suppress the niche cluster loadings visualization.
    --suppress-niche-trajectory
                        Suppress the niche trajectory related visualization.
```

### Full parameters for ONTraC_GP

```{text}
Usage: ONTraC_GP <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>
    [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--trajectory-construct TRAJECTORY_CONSTRUCT]

ONTraC_GP: GNN and Niche Trajectory

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit

  IO:
    --preprocessing-dir=PREPROCESSING_DIR
                        Directory for preprocessing outputs.
    --GNN-dir=GNN_DIR   Directory for the GNN output.
    --NTScore-dir=NTSCORE_DIR
                        Directory for the NTScore output.
    -d DATASET, --dataset=DATASET
                        Original input dataset.

  Options for GNN training:
    --device=DEVICE     Device for training. We support cpu and cuda now. Auto select if not specified.
    --epochs=EPOCHS     Number of maximum epochs for training. Default is 1000.
    --patience=PATIENCE
                        Number of epochs wait for better result. Default is 100.
    --min-delta=MIN_DELTA
                        Minimum delta for better result. Default is 0.001
    --min-epochs=MIN_EPOCHS
                        Minimum number of epochs for training. Default is 50. Set to 0 to disable.
    --batch-size=BATCH_SIZE
                        Batch size for training. Default is 0 for whole dataset.
    -s SEED, --seed=SEED
                        Random seed for training. Default is random.
    --lr=LR             Learning rate for training. Default is 0.03.
    --hidden-feats=HIDDEN_FEATS
                        Number of hidden features. Default is 4.
    -k K, --k-clusters=K
                        Number of niche clusters. Default is 6.
    --modularity-loss-weight=MODULARITY_LOSS_WEIGHT
                        Weight for modularity loss. Default is 0.3.
    --purity-loss-weight=PURITY_LOSS_WEIGHT
                        Weight for purity loss. Default is 300.
    --regularization-loss-weight=REGULARIZATION_LOSS_WEIGHT
                        Weight for regularization loss. Default is 0.1.
    --beta=BETA         Beta value control niche cluster assignment matrix. Default is 0.03.

  Options for niche trajectory:
    --trajectory-construct=TRAJECTORY_CONSTRUCT
                        Method to construct the niche trajectory. Default is
                        'BF' (brute-force). A faster alternative is 'TSP'.
```

## Detailed explanation

A detailed explanation for some parameters is listed below.

### patience

The training process will terminated if the model does not improve after the number of epochs set by this parameter.

### min-delta

The model will be considered improved if the total loss decreases by the propotion set by this parameter.

### hidden-feats

The number of niche features after GCN processing (step2).

### modularity-loss-weight

The modularity loss controls the spatial smoothness of niche clusters.

### purity-loss-weight

The purity loss controls the purity of cell type composition within each niche cluster.

### regularization-loss-weight

The regularization loss controls the balance of niche cluster sizes. The higher this weight is set, the more equal the size of each niche cluster.

### beta

The beta value of the softmax function used in generating niche cluster assignment matrix.
