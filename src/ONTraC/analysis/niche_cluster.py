from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ..log import warning
from .data import AnaData
from .utils import gini, saptial_figsize


def plot_niche_cluster_connectivity(
    niche_cluster_connectivity: np.ndarray,
    niche_cluster_score: np.ndarray,
    reverse: bool = False,
    output_file_path: Optional[Union[str,
                                     Path]] = None) -> Optional[Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]:
    """
    Plot niche cluster connectivity.
    :param niche_cluster_connectivity: np.ndarray, the connectivity matrix.
    :param niche_cluster_score: Optional[np.ndarray], the score of each niche cluster.
    :param reverse: bool, whether to reverse the color.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    G = nx.Graph(niche_cluster_connectivity / niche_cluster_connectivity.max())

    # position
    pos = nx.spring_layout(G, seed=42)
    # edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # node color
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
    nc_scores = 1 - niche_cluster_score if reverse else niche_cluster_score
    niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in G.nodes]

    # Create a figure
    fig = plt.figure(figsize=(7, 6))

    # Create a gridspec with 1 row and 2 columns, with widths of A and B
    gs = gridspec.GridSpec(1, 3, width_ratios=[6, .5, .5])  # 6:.5:.5 ratio
    graph_ax = fig.add_subplot(gs[0])
    nodel_colorbar_ax = fig.add_subplot(gs[1])
    edge_colorbar_ax = fig.add_subplot(gs[2])

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=niche_cluster_colors, node_size=500, ax=graph_ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=weights,
        alpha=weights,
        width=3.0,
        edge_cmap=plt.cm.Reds,  # type: ignore
        ax=graph_ax)
    nx.draw_networkx_labels(G, pos, ax=graph_ax)
    graph_ax.set_title('Niche cluster connectivity')

    # Draw the colorbar for nodes
    gradient = np.linspace(0, 1, 1000).reshape(-1, 1)
    nodel_colorbar_ax.imshow(gradient, aspect='auto', cmap=plt.cm.rainbow)
    nodel_colorbar_ax.set_xticks([])
    nodel_colorbar_ax.set_yticks(np.linspace(0, 1000, 5))
    nodel_colorbar_ax.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, 1, 5))
    nodel_colorbar_ax.set_ylabel('NT score (nodes)')

    # Draw the colorbar for edges
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    gradient = np.linspace(1, 0, 1000).reshape(-1, 1)
    edge_colorbar_ax.imshow(gradient, aspect='auto', cmap=custom_cmap)
    edge_colorbar_ax.set_xticks([])
    edge_colorbar_ax.set_yticks(np.linspace(1000, 0, 5))
    edge_colorbar_ax.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, niche_cluster_connectivity.max(), 5))
    edge_colorbar_ax.set_ylabel('Connectivity (edges)')

    fig.tight_layout()

    if output_file_path is not None:
        fig.savefig(output_file_path)
        plt.close(fig)
        return None
    else:
        return fig, [graph_ax, nodel_colorbar_ax, edge_colorbar_ax]


def plot_niche_cluster_connectivity_from_anadata(
        ana_data: AnaData) -> Optional[Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]:
    """
    Plot niche cluster connectivity.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    try:
        if ana_data.niche_cluster_score is None or ana_data.niche_cluster_connectivity is None:
            warning("No niche cluster score or connectivity data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    output_file_path = f'{ana_data.options.output}/niche_cluster_connectivity.pdf' if ana_data.options.output is not None else None
    return plot_niche_cluster_connectivity(niche_cluster_connectivity=ana_data.niche_cluster_connectivity,
                                           niche_cluster_score=ana_data.niche_cluster_score,
                                           reverse=ana_data.options.reverse,
                                           output_file_path=output_file_path)


def plot_niche_cluster_connectivity_bysample_from_anadata(ana_data: AnaData) -> None:
    """
    Plot niche cluster connectivity by sample.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    for sample in ana_data.cell_type_composition['sample'].unique():
        niche_cluster_connectivity = np.loadtxt(f'{ana_data.options.GNN_dir}/{sample}_out_adj.csv.gz', delimiter=',')
        output_file_path = f'{ana_data.options.output}/{sample}_cluster_connectivity.pdf'
        plot_niche_cluster_connectivity(niche_cluster_connectivity=niche_cluster_connectivity,
                                        niche_cluster_score=ana_data.niche_cluster_score,
                                        reverse=ana_data.options.reverse,
                                        output_file_path=output_file_path)


def plot_cluster_proportion(
        niche_cluster_loading: pd.DataFrame,
        niche_cluster_colors: List,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the proportion of each cluster.
    :param niche_cluster_loading: pd.DataFrame, the loading of each niche cluster.
    :param niche_cluster_colors: List, the color of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(niche_cluster_loading,
           labels=[f'Niche cluster {i}' for i in range(niche_cluster_loading.shape[0])],
           colors=niche_cluster_colors,
           autopct='%1.1f%%',
           pctdistance=1.25,
           labeldistance=.6)
    ax.set_title(f'Niche proportions for each niche cluster')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/niche_cluster_proportion.pdf')
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cluster_proportion_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the proportion of each cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    try:
        if ana_data.niche_cluster_score is None or ana_data.niche_level_niche_cluster_assign is None:
            warning("No niche cluster score or connectivity data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    # colors
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
    nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
    niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in np.arange(ana_data.niche_cluster_score.shape[0])]

    # loadings
    niche_cluster_loading = ana_data.niche_level_niche_cluster_assign.sum(axis=0)

    return plot_cluster_proportion(niche_cluster_loading=niche_cluster_loading,
                                   niche_cluster_colors=niche_cluster_colors,
                                   output_file_path=ana_data.options.output)


def plot_niche_cluster_loadings_dataset(
    cell_level_niche_cluster_assign: pd.DataFrame,
    cell_id: pd.DataFrame,
    nc_scores: np.ndarray,
    output_file_path: Optional[Union[str,
                                     Path]] = None) -> Optional[Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]:
    """
    Plot niche cluster loadings for each cell.
    :param cell_level_niche_cluster_assign: pd.DataFrame, the niche cluster assign data.
    :param cell_id: pd.DataFrame, the cell id data.
    :param nc_scores: np.ndarray, the score of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    samples = cell_id['Sample'].unique()
    n_sample = len(samples)
    n_niche_cluster = cell_level_niche_cluster_assign.shape[1]

    fig, axes = plt.subplots(n_sample, n_niche_cluster, figsize=(3.3 * n_niche_cluster, 3 * n_sample))
    for i, sample in enumerate(samples):
        sample_df = cell_level_niche_cluster_assign.loc[cell_id[cell_id['Sample'] == sample].index]
        sample_df = sample_df.join(cell_id[['x', 'y']])
        for j, c_index in enumerate(nc_scores.argsort()):
            ax = axes[i, j] if n_sample > 1 else axes[j]
            scatter = ax.scatter(sample_df['x'],
                                 sample_df['y'],
                                 c=sample_df[f'NicheCluster_{c_index}'],
                                 cmap='Reds',
                                 vmin=0,
                                 vmax=1,
                                 s=4)
            ax.set_title(f'{sample}: niche cluster {c_index}')
            plt.colorbar(scatter)
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/niche_cluster_loadings.pdf')
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_niche_cluster_loadings_dataset_from_anadata(
        ana_data: AnaData) -> Optional[Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]:
    """
    Plot niche cluster loadings for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]].
    """

    try:
        if ana_data.niche_cluster_score is None or ana_data.cell_type_composition is None or ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster score, cell type composition or cluster assign data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score

    return plot_niche_cluster_loadings_dataset(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                                               cell_id=ana_data.cell_id,
                                               nc_scores=nc_scores,
                                               output_file_path=ana_data.options.output)


def plot_niche_cluster_loadings_sample(
        cell_level_niche_cluster_assign: pd.DataFrame,
        cell_id: pd.DataFrame,
        nc_scores: np.ndarray,
        spatial_scaling_factor: float = 1.0,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot niche cluster loadings for each cell.
    :param cell_level_niche_cluster_assign: pd.DataFrame, the niche cluster assign data.
    :param cell_id: pd.DataFrame, the cell id data.
    :param nc_scores: np.ndarray, the score of each niche cluster.
    :param spatial_scaling_factor: float, the scale factor control the size of spatial-based plots.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    samples = cell_id['Sample'].unique()

    output = []
    for sample in samples:
        sample_df = cell_level_niche_cluster_assign.loc[cell_id[cell_id['Sample'] == sample].index]
        sample_df = sample_df.join(cell_id[['x', 'y']])
        fig_width, fig_height = saptial_figsize(sample_df, scaling_factor=spatial_scaling_factor)
        fig, axes = plt.subplots(1, nc_scores.shape[0], figsize=(fig_width * nc_scores.shape[0], fig_height))
        for j, c_index in enumerate(nc_scores.argsort()):
            ax = axes[j]  #  there should more than one niche cluster
            scatter = ax.scatter(sample_df['x'],
                                 sample_df['y'],
                                 c=sample_df[f'NicheCluster_{c_index}'],
                                 cmap='Reds',
                                 vmin=0,
                                 vmax=1,
                                 s=4)
            ax.set_title(f'{sample}: niche cluster {c_index}')
            plt.colorbar(scatter)
        fig.tight_layout()
        output.append((fig, axes))
        if output_file_path is not None:
            fig.savefig(f'{output_file_path}/niche_cluster_loadings_{sample}.pdf')
            plt.close(fig)
    return output if len(output) > 0 else None


def plot_niche_cluster_loadings_sample_from_anadata(ana_data: AnaData) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot niche cluster loadings for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    try:
        if ana_data.niche_cluster_score is None or ana_data.cell_type_composition is None or ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster score, cell type composition or cluster assign data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score

    return plot_niche_cluster_loadings_sample(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                                              cell_id=ana_data.cell_id,
                                              nc_scores=nc_scores,
                                              spatial_scaling_factor=ana_data.options.scale_factor,
                                              output_file_path=ana_data.options.output)


def plot_niche_cluster_loadings(
    ana_data: AnaData
) -> Optional[Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]]:
    """
    Plot niche cluster loadings for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]].
    """
    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        return plot_niche_cluster_loadings_sample_from_anadata(ana_data=ana_data)
    else:
        return plot_niche_cluster_loadings_dataset_from_anadata(ana_data=ana_data)


def plot_max_niche_cluster_dataset(
        cell_level_max_niche_cluster: pd.DataFrame,
        cell_id: pd.DataFrame,
        nc_scores: np.ndarray,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the maximum niche cluster for each cell.
    :param cell_level_max_niche_cluster: pd.DataFrame, the maximum niche cluster data.
    :param cell_id: pd.DataFrame, the cell id data.
    :param nc_scores: np.ndarray, the score of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    samples = cell_id['Sample'].unique()
    n_sample = len(samples)

    # colors
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
    niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in np.arange(cell_level_max_niche_cluster.shape[1])]
    palette = {f'niche cluster {i}': niche_cluster_colors[i] for i in range(cell_level_max_niche_cluster.shape[1])}

    fig, axes = plt.subplots(1, n_sample, figsize=(5 * n_sample, 3))
    for i, sample in enumerate(samples):
        ax: Axes = axes[i] if n_sample > 1 else axes  # type: ignore
        sample_df = cell_level_max_niche_cluster.loc[cell_id[cell_id['sample'] == sample].index]
        sample_df = sample_df.join(cell_id[['x', 'y']])
        sample_df['Niche_Cluster'] = 'niche cluster ' + sample_df['Niche_Cluster'].astype(str)
        sns.scatterplot(data=sample_df,
                        x='x',
                        y='y',
                        hue='Niche_Cluster',
                        hue_order=[f'niche cluster {j}' for j in nc_scores.argsort()],
                        palette=palette,
                        s=10,
                        ax=ax)
        ax.set_title(f'{sample}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/max_niche_cluster.pdf')
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_max_niche_cluster_dataset_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the maximum niche cluster for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if ana_data.niche_cluster_score is None or ana_data.cell_type_composition is None or ana_data.cell_level_max_niche_cluster is None:
            warning("No niche cluster score, cell type composition or max niche cluster data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score

    return plot_max_niche_cluster_dataset(cell_level_max_niche_cluster=ana_data.cell_level_max_niche_cluster,
                                          cell_id=ana_data.cell_id,
                                          nc_scores=nc_scores,
                                          output_file_path=ana_data.options.output)


def plot_max_niche_cluster_sample_from_anadata(ana_data: AnaData) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot the maximum niche cluster for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    try:
        if ana_data.niche_cluster_score is None or ana_data.cell_type_composition is None or ana_data.cell_level_max_niche_cluster is None:
            warning("No niche cluster score, cell type composition or max niche cluster data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    # colors
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
    nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
    niche_cluster_colors = [sm.to_rgba(nc_scores[n]) for n in np.arange(ana_data.niche_cluster_score.shape[0])]
    palette = {f'niche cluster {i}': niche_cluster_colors[i] for i in range(ana_data.niche_cluster_score.shape[0])}
    samples: List[str] = ana_data.cell_type_composition['sample'].unique().tolist()

    output = []
    for sample in samples:
        sample_df = ana_data.cell_level_max_niche_cluster.loc[ana_data.cell_type_composition[
            ana_data.cell_type_composition['sample'] == sample].index]
        sample_df = sample_df.join(ana_data.cell_type_composition[['x', 'y']])
        sample_df['Niche_Cluster'] = 'niche cluster ' + sample_df['Niche_Cluster'].astype(str)
        fig_width, fig_height = saptial_figsize(sample_df, scaling_factor=ana_data.options.scale_factor)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        sns.scatterplot(data=sample_df,
                        x='x',
                        y='y',
                        hue='Niche_Cluster',
                        hue_order=[f'niche cluster {j}' for j in nc_scores.argsort()],
                        palette=palette,
                        s=10,
                        ax=ax)
        ax.set_title(f'{sample}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        output.append((fig, ax))
        if ana_data.options.output is not None:
            fig.savefig(f'{ana_data.options.output}/max_niche_cluster_{sample}.pdf')
            plt.close(fig)
    return output if len(output) > 0 else None


def plot_max_niche_cluster(
        ana_data: AnaData) -> Optional[Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot the maximum niche cluster for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        return plot_max_niche_cluster_sample_from_anadata(ana_data=ana_data)
    else:
        return plot_max_niche_cluster_dataset_from_anadata(ana_data=ana_data)


def plot_niche_cluster_gini(
        intra_cluster_gini_df: pd.DataFrame,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the Gini coefficient of each niche cluster.
    :param intra_cluster_gini_df: pd.DataFrame, the Gini coefficient of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=intra_cluster_gini_df, x='cluster', y='gini', ax=ax)
    ax.set_xlabel('Niche Cluster')
    ax.set_ylabel('Gini coefficient across each cell')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/niche_cluster_gini.pdf')
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_niche_cluster_gini_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the Gini coefficient of each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    try:
        if ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster assign data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None
    intra_cluster_gini = ana_data.cell_level_niche_cluster_assign.apply(gini, axis=0).values
    intra_cluster_gini_df = pd.DataFrame(data={
        'gini': intra_cluster_gini,
        'cluster': ana_data.cell_level_niche_cluster_assign.columns
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=intra_cluster_gini_df, x='cluster', y='gini', ax=ax)
    ax.set_xlabel('Niche Cluster')
    ax.set_ylabel('Gini coefficient across each cell')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/niche_cluster_gini.pdf')
        plt.close(fig)
        return None
    else:
        return fig, ax


def niche_cluster_visualization(ana_data: AnaData) -> None:
    """
    All spatial visualization will include here.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    # 1. plot niche cluster connectivity
    plot_niche_cluster_connectivity_from_anadata(ana_data=ana_data)

    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        plot_niche_cluster_connectivity_bysample_from_anadata(ana_data=ana_data)

    # 2. share of each cluster
    plot_cluster_proportion_from_anadata(ana_data=ana_data)

    # 3. niche cluster loadings for each cell
    if not hasattr(ana_data.options,
                   'suppress_niche_cluster_loadings') or not ana_data.options.suppress_niche_cluster_loadings:
        plot_niche_cluster_loadings(ana_data=ana_data)

    # 4. maximum niche cluster for each cell
    plot_max_niche_cluster(ana_data=ana_data)

    # 5. gini coefficient of each niche cluster
    plot_niche_cluster_gini_from_anadata(ana_data=ana_data)
