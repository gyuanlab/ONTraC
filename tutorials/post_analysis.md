# Post-analysis

Below is an example of post-analysis on stereo-seq brain data.
The command for running on this dataset is:

```{sh}
ONTraC -d stereo_seq_dataset.csv --preprocessing-dir stereo_seq_final_preprocessing_dir --GNN-dir stereo_seq_final_GNN --NTScore-dir stereo_seq_final_NTScore --epochs 100 --batch-size 5 -s 42 --patience 100 --min-delta 0.001 --min-epochs 50 --lr 0.03 --hidden-feats 4 -k 6 --modularity-loss-weight 0.3 --regularization-loss-weight 0.1 --purity-loss-weight 300 --beta 0.03 2>&1 | tee stereo_seq_final.log
```

The input dataset and output files could be downloaded from [Zenodo](https://zenodo.org/records/11186620).

## prepare

### Install required packages

```{sh}
pip install ONTraC[analysis]
# or
pip install seaborn
```

## One line command

You can get all the output figures with this command and check the results in `analysis_output` directory.

```{sh}
ONTraC_analysis -o analysis_output -l ONTraC.log --preprocessing-dir preprocessing_dir --GNN-dir GNN_dir --NTScore-dir NTScore_dir
```

## Step-by-step analysis

If you want to adjust the figures, here is the example codes for step-by-step analysis using Python.
We recommand you using jupyter notebook or jupyter lab here.

### Load modules

```{python}
import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ONTraC.analysis.data import AnaData
```

### Plotting preprare

```{python}
from optparse import Values

options = Values()
options.dataset = 'stereo_seq_dataset.csv'
options.preprocessing_dir = 'stereo_seq_final_preprocessing_dir'
options.GNN_dir = 'stereo_seq_final_GNN'
options.NTScore_dir = 'stereo_seq_final_NTScore'
options.log = 'stereo_seq_final.log'
options.reverse = True  # Set it to False if you don't want reverse NT score
ana_data = AnaData(options)
```

### Spatial cell type distribution

```{python}
data_df = ana_data.cell_id.join(ana_data.cell_type_composition[['sample', 'x', 'y']])
samples = data_df['sample'].unique()
N = len(samples)
fig, axes = plt.subplots(1, N, figsize = (4 * N, 3))
for i, sample in enumerate(samples):
    sample_df = data_df.loc[data_df['sample'] == sample]
    ax = axes[i] if N > 1 else axes
    sns.scatterplot(data = sample_df,
                x = 'x',
                y = 'y',
                hue = 'Cell_Type',
                hue_order = ['RGC', 'GlioB', 'NeuB', 'GluNeuB', 'GluNeu', 'GABA', 'Ery', 'Endo', 'Fibro', 'Basal'],  # change based on your own dataset or remove this line
                s = 8,
                ax = ax)
    # ax.set_aspect('equal', 'box')  # uncomment this line if you want set the x and y axis with same scaling
    # ax.set_xticks([])  # uncomment this line if you don't want to show x coordinates
    # ax.set_yticks([]) # uncomment this line if you don't want to show y coordinates
    ax.set_title(f"{sample}")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))


fig.tight_layout()
fig.savefig('figures/Spatial_cell_type.png', dpi=150)
```

![spatial_cell_type_image](../docs/source/_static/images/tutorials/post_analysis/Spatial_cell_type.png)

### Spatial cell type composition distribution

```{python}
samples = ana_data.cell_type_composition['sample'].unique().tolist()
cell_types = ana_data.cell_type_codes['Cell_Type'].tolist()

M, N = len(samples), len(cell_types)
fig, axes = plt.subplots(M, N, figsize = (3.5 * N, 3 * M))
for i, sample in enumerate(samples):
    sample_df = ana_data.cell_type_composition.loc[ana_data.cell_type_composition['sample'] == sample]
    for j, cell_type in enumerate(cell_types):
        ax = axes[i, j] if M > 1 else axes[j]
        scatter = ax.scatter(sample_df['x'], sample_df['y'], c=sample_df[cell_type], cmap='Reds', vmin=0, vmax=1, s=1)
        # ax.set_aspect('equal', 'box')  # uncomment this line if you want set the x and y axis with same scaling
        # ax.set_xticks([])  # uncomment this line if you don't want to show x coordinates
        # ax.set_yticks([]) # uncomment this line if you don't want to show y coordinates
        plt.colorbar(scatter)
        ax.set_title(f"{sample} {cell_type}")


fig.tight_layout()
fig.savefig('figures/cell_type_compostion.png', dpi=100)
```

![cell_type_composition_image](../docs/source/_static/images/tutorials/post_analysis/cell_type_compostion.png)

### Niche cluster

#### Spatial niche cluster loadings distribution

```{python}
nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
samples = ana_data.cell_type_composition['sample'].unique().tolist()
M, N = len(samples), ana_data.cell_level_niche_cluster_assign.shape[1]

fig, axes = plt.subplots(M, N, figsize=(3.3 * N, 3 * M))
for i, sample in enumerate(samples):
    sample_df = ana_data.cell_level_niche_cluster_assign.loc[ana_data.cell_type_composition[
        ana_data.cell_type_composition['sample'] == sample].index]
    sample_df = sample_df.join(ana_data.cell_type_composition[['x', 'y']])
    for j, c_index in enumerate(nc_scores.argsort()):
        ax = axes[i, j] if M > 1 else axes[j]
        scatter = ax.scatter(sample_df['x'],
                             sample_df['y'],
                             c=sample_df[f'NicheCluster_{c_index}'],
                             cmap='Reds',
                             vmin=0,
                             vmax=1,
                             s=4)
        ax.set_title(f'{sample}: niche cluster {c_index}')
        # ax.set_aspect('equal', 'box')  # uncomment this line if you want set the x and y axis with same scaling
        # ax.set_xticks([])  # uncomment this line if you don't want to show x coordinates
        # ax.set_yticks([]) # uncomment this line if you don't want to show y coordinates
        plt.colorbar(scatter)
fig.tight_layout()
fig.savefig('figures/Spatial_niche_clustering_loadings.png', dpi=100)
```

![spatial_niche_cluster_loadings_image](../docs/source/_static/images/tutorials/post_analysis/Spatial_niche_clustering_loadings.png)

#### Spatial maximum niche cluster distribution

```{python}
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Set the colors corresponding to NT score
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)
nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in np.arange(ana_data.niche_cluster_score.shape[0])]
palette = {f'niche cluster {i}': niche_cluster_colors[i] for i in range(ana_data.niche_cluster_score.shape[0])}

samples = ana_data.cell_type_composition['sample'].unique().tolist()
M = len(samples)

fig, axes = plt.subplots(1, M, figsize=(5 * M, 3))
for i, sample in enumerate(samples):
    ax = axes[i] if M > 1 else axes
    sample_df = ana_data.cell_level_max_niche_cluster.loc[ana_data.cell_type_composition[
        ana_data.cell_type_composition['sample'] == sample].index]
    sample_df = sample_df.join(ana_data.cell_type_composition[['x', 'y']])
    sample_df['Niche_Cluster'] = 'niche cluster ' + sample_df['Niche_Cluster'].astype(str)
    sns.scatterplot(data=sample_df,
                    x='x',
                    y='y',
                    hue='Niche_Cluster',
                    hue_order=[f'niche cluster {j}' for j in nc_scores.argsort()],
                    palette=palette,  # Comment this line if you don't want colors corresponding to NT score
                    s=10,
                    ax=ax)
    ax.set_title(f'{sample}')
    # ax.set_aspect('equal', 'box')  # uncomment this line if you want set the x and y axis with same scaling
    # ax.set_xticks([])  # uncomment this line if you don't want to show x coordinates
    # ax.set_yticks([]) # uncomment this line if you don't want to show y coordinates
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig('figures/Spatial_max_niche_cluster.png', dpi=300)
```

![spatial_max_niche_cluster_image](../docs/source/_static/images/tutorials/post_analysis/Spatial_max_niche_cluster.png)

#### Niche cluster connectivity

```{python}
import networkx as nx
import matplotlib.gridspec as gridspec

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


G = nx.Graph(ana_data.niche_cluster_connectivity)

# position
pos = nx.spring_layout(G, seed=42)
# edges
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
# Set the colors corresponding to NT score
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in G.nodes]

# Create a figure
fig = plt.figure(figsize=(7, 6))

# Create a gridspec with 1 row and 2 columns, with widths of A and B
gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])  # 6:1 ratio
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_color=niche_cluster_colors, node_size=500, ax=ax1)
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=weights,
    alpha=weights,
    width=3.0,
    edge_cmap=plt.cm.Reds,  # type: ignore
    ax=ax1)
nx.draw_networkx_labels(G, pos, ax=ax1)
ax1.set_title('Niche cluster connectivity')

# Draw the colorbar
colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
gradient = np.linspace(1, 0, 1000).reshape(-1, 1)
ax2.imshow(gradient, aspect='auto', cmap=custom_cmap)
ax2.set_xticks([])
ax2.set_yticks(np.linspace(1000, 0, 5))
ax2.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, ana_data.niche_cluster_connectivity.max(), 5))
ax2.set_ylabel('Connectivity')

fig.tight_layout()

fig.savefig('figures/Niche_cluster_connectivity.png', dpi=300)
```

![niche_cluster_connectivity_image](../docs/source/_static/images/tutorials/post_analysis/Niche_cluster_connectivity.png)

#### Niche cluster proportion

```{python}
# Set the colors corresponding to NT score
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in np.arange(ana_data.niche_cluster_score.shape[0])]

# loadings
niche_cluster_loading = ana_data.niche_level_niche_cluster_assign.sum(axis=0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(niche_cluster_loading,
       labels=[f'Niche cluster {i}' for i in range(niche_cluster_loading.shape[0])],
       colors=niche_cluster_colors,  # Comment this line if you don't want colors corresponding to NT score
       autopct='%1.1f%%',
       pctdistance=1.25,
       labeldistance=.6)
ax.set_title(f'Niche proportions for each niche cluster')
fig.tight_layout()
fig.savefig('figures/Niche_cluster_proportions.png', dpi=300)
```

![niche_cluster_proportions_image](../docs/source/_static/images/tutorials/post_analysis/Niche_cluster_proportions.png)

### Cell type distribution in each niche cluster

#### calculate cell type distribution in each niche cluster

```{python}
data_df = ana_data.cell_id.join(ana_data.cell_level_niche_cluster_assign)
t = pd.CategoricalDtype(categories=ana_data.cell_type_codes['Cell_Type'], ordered=True)
cell_type_one_hot = np.zeros(shape=(data_df.shape[0], ana_data.cell_type_codes.shape[0]))
cell_type = data_df['Cell_Type'].astype(t)
cell_type_one_hot[np.arange(data_df.shape[0]), cell_type.cat.codes] = 1  # N x n_cell_type
cell_type_dis = np.matmul(data_df[ana_data.cell_level_niche_cluster_assign.columns].T,
                          cell_type_one_hot)  # n_clusters x n_cell_types
cell_type_dis_df = pd.DataFrame(cell_type_dis)
cell_type_dis_df.columns = ana_data.cell_type_codes['Cell_Type']
# nc_order
nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
nc_order = [f'NicheCluster_{x}' for x in nc_scores.argsort()]
cell_type_dis_df = cell_type_dis_df.loc[nc_order]
```

#### number of cells of each cell type cells in each niche cluster

```{python}
from copy import deepcopy

data_df = deepcopy(cell_type_dis_df)
cell_type = data_df.columns
data_df['cluster'] = data_df.index
cell_type_dis_melt_df = pd.melt(
    data_df,
    id_vars='cluster',
    var_name='Cell type',
    value_vars=cell_type,
    value_name='Number')
g = sns.catplot(cell_type_dis_melt_df, kind="bar", x="Number", y="Cell type", col="cluster", height=2 + len(cell_type) / 6,
                aspect=.5)
g.add_legend()
g.tight_layout()
g.set_xticklabels(rotation='vertical')
g.savefig('figures/cell_type_loading_in_niche_clusters.png', dpi=300)
```

![cell_type_loading_in_niche_clusters_image](../docs/source/_static/images/tutorials/post_analysis/cell_type_loading_in_niche_clusters.png)

#### cell type proportions in each cluster normalized by total loadings of each niche cluster

```{python}
fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 4))
sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=1), ax=ax)
ax.set_xlabel('Cell Type')
ax.set_ylabel('Niche Cluster')
fig.tight_layout()
fig.savefig('figures/cell_type_dis_in_niche_clusters.png', dpi=300)
```

![cell_type_dis_in_niche_clusters_image](../docs/source/_static/images/tutorials/post_analysis/cell_type_dis_in_niche_clusters.png)

#### cell type proportions in each cluster normalized by the number of each cell type

```{python}
fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 4))
sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=0), ax=ax)
ax.set_xlabel('Cell Type')
ax.set_ylabel('Niche Cluster')
fig.tight_layout()
fig.savefig('figures/cell_type_dis_across_niche_clusters.png', dpi=300)
```

![cell_type_dis_across_niche_clusters_image](../docs/source/_static/images/tutorials/post_analysis/cell_type_dis_across_niche_clusters.png)

### Spatial niche-level NT score distribution

```{python}
samples = ana_data.NT_score['sample'].unique().tolist()

N = len(samples)
fig, axes = plt.subplots(1, N, figsize=(3.5 * N, 3))
for i, sample in enumerate(samples):
    sample_df = ana_data.NT_score.loc[ana_data.NT_score['sample'] == sample]
    ax = axes[i] if N > 1 else axes
    NT_score = sample_df['Niche_NTScore'] if not ana_data.options.reverse else 1 - sample_df['Niche_NTScore']
    scatter = ax.scatter(sample_df['x'], sample_df['y'], c=NT_score, cmap='rainbow', vmin=0, vmax=1, s=1)
    # ax.set_aspect('equal', 'box')  # uncomment this line if you want set the x and y axis with same scaling
    # ax.set_xticks([])  # uncomment this line if you don't want to show x coordinates
    # ax.set_yticks([]) # uncomment this line if you don't want to show y coordinates
    plt.colorbar(scatter)
    ax.set_title(f"{sample} Niche-level NT Score")

fig.tight_layout()
fig.savefig('figures/niche_NT_score.png', dpi=200)
```

![niche_level_NT_score_image](../docs/source/_static/images/tutorials/post_analysis/niche_NT_score.png)

## Spatial cell-level NT score distribution

```{python}
samples = ana_data.NT_score['sample'].unique().tolist()

N = len(samples)
fig, axes = plt.subplots(1, N, figsize=(3.5 * N, 3))
for i, sample in enumerate(samples):
    sample_df = ana_data.NT_score.loc[ana_data.NT_score['sample'] == sample]
    ax = axes[i] if N > 1 else axes
    NT_score = sample_df['Cell_NTScore'] if not ana_data.options.reverse else 1 - sample_df['Cell_NTScore']
    scatter = ax.scatter(sample_df['x'], sample_df['y'], c=NT_score, cmap='rainbow', vmin=0, vmax=1, s=1)
    # ax.set_aspect('equal', 'box')  # uncomment this line if you want set the x and y axis with same scaling
    # ax.set_xticks([])  # uncomment this line if you don't want to show x coordinates
    # ax.set_yticks([]) # uncomment this line if you don't want to show y coordinates
    plt.colorbar(scatter)
    ax.set_title(f"{sample} Cell-level NT Score")

fig.tight_layout()
fig.savefig('figures/cell_NT_score.png', dpi=200)
```

![cell_level_NT_score_image](../docs/source/_static/images/tutorials/post_analysis/cell_NT_score.png)

## Cell-level NT score distribution for each cell type

```{python}
data_df = ana_data.cell_id.join(ana_data.NT_score['Cell_NTScore'])
if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

fig, ax = plt.subplots(figsize=(6, ana_data.cell_type_codes.shape[0] / 2))
sns.violinplot(data=data_df,
               x='Cell_NTScore',
               y='Cell_Type',
               order = ['RGC', 'GlioB', 'NeuB', 'GluNeuB', 'GluNeu', 'GABA', 'Ery', 'Endo', 'Fibro', 'Basal'],  # change based on your own dataset or remove this line
               hue='Cell_Type',
               cut=0,
               fill=False,
               common_norm=True,
               legend=False,
               ax=ax)
ax.set_xlabel('Cell-level NT score')
ax.set_ylabel('Cell Type')
fig.tight_layout()
fig.savefig('figures/cell_type_along_NT_score_violin.png', dpi=300)
```

![cell_level_NT_score_distribution_for_each_cell_type](../docs/source/_static/images/tutorials/post_analysis/cell_type_along_NT_score_violin.png)
