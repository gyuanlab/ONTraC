import os
from optparse import Values
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import load_npz

from ..data import SpatailOmicsDataset
from ..log import error, info
from .algorithm import brute_force, held_karp


def load_consolidate_data(options: Values) -> Tuple[ndarray, ndarray]:
    """
    Load consolidate s_array and out_adj_array
    :param options: Values, options
    :return: Tuple[ndarray, ndarray], the consolidate s_array and out_adj_array
    """

    info('Loading consolidate s_array and out_adj_array...')

    if not os.path.exists(f'{options.GNN_dir}/consolidate_s.csv.gz') or not os.path.exists(
            f'{options.GNN_dir}/consolidate_out_adj.csv.gz'):
        error(f'consolidate_s.csv.gz or consolidate_out_adj.csv.gz does not exist in {options.GNN_dir} directory.')
    consolidate_s_array = np.loadtxt(fname=f'{options.GNN_dir}/consolidate_s.csv.gz', delimiter=',')
    consolidate_out_adj_array = np.loadtxt(fname=f'{options.GNN_dir}/consolidate_out_adj.csv.gz', delimiter=',')

    return consolidate_s_array, consolidate_out_adj_array


def get_niche_trajectory_path(options: Values, niche_adj_matrix: ndarray) -> List[int]:
    """
    Find niche level trajectory with maximum connectivity using Brute Force
    :param adj_matrix: non-negative ndarray, adjacency matrix of the graph
    :param methods: str, the method to find the niche trajectory path. Default is 'BF' (Brute Force).
    :return: List[int], the niche trajectory
    """
    niche_adj_matrix = (niche_adj_matrix + niche_adj_matrix.T) / 2
    
    if options.trajectory_construct == 'BF':
        info('Finding niche trajectory with maximum connectivity using Brute Force.')

        niche_trajectory_path = brute_force(niche_adj_matrix)

    elif options.trajectory_construct == 'TSP':
        info('Finding niche trajectory with maximum connectivity using TSP.')

        niche_trajectory_path = held_karp(niche_adj_matrix)

    return niche_trajectory_path


def trajectory_path_to_NC_score(niche_trajectory_path: List[int]) -> ndarray:
    """
    Convert niche cluster trajectory path to NTScore
    :param niche_trajectory_path: List[int], the niche trajectory path
    :return: ndarray, the NTScore
    """

    info('Calculating NTScore for each niche cluster based on the trajectory path.')

    niche_NT_score = np.zeros(len(niche_trajectory_path))
    values = np.linspace(0, 1, len(niche_trajectory_path))

    for i, index in enumerate(niche_trajectory_path):
        # debug(f'i: {i}, index: {index}')
        niche_NT_score[index] = values[i]
    return niche_NT_score


def apply_diffusion_map(niche_adj_matrix: ndarray) -> ndarray:
    """
    Apply diffusion map to the niche adjacency matrix
    :param niche_adj_matrix: ndarray, the adjacency matrix of the graph
    :return: ndarray, the NTScore
    """

    info('Applying diffusion map to the niche adjacency matrix.')

    # niche_adj_matrix = niche_adj_matrix + niche_adj_matrix.T
    D = np.diag(np.power(niche_adj_matrix.sum(axis=1), -0.5))
    L = D @ niche_adj_matrix @ D
    eigvals, eigvecs = np.linalg.eig(L)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    niche_cluster_score = eigvecs[:, 1]
    # normalize to [0, 1]
    # niche_cluster_score -= niche_cluster_score.min()
    # niche_cluster_score /= niche_cluster_score.max()
    # normalize to [-1, 1] by rank
    niche_cluster_score = niche_cluster_score.argsort().argsort()
    niche_cluster_score = niche_cluster_score / (niche_cluster_score.size - 1)
    return niche_cluster_score


def get_niche_NTScore(options: Values, niche_cluster_loading: ndarray,
                      niche_adj_matrix: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Get niche-level niche trajectory and cell-level niche trajectory
    :param niche_cluster_loading: ndarray, the loading of cell x niche clusters
    :param adj_matrix: ndarray, the adjacency matrix of the graph
    :return: Tuple[ndarray, ndarray], the niche-level niche trajectory and cell-level niche trajectory
    """

    info('Calculating NTScore for each niche.')

    if options.trajectory_construct == 'DM':
        niche_cluster_score = apply_diffusion_map(niche_adj_matrix)
    else:
        niche_trajectory_path = get_niche_trajectory_path(options=options, niche_adj_matrix=niche_adj_matrix)
        niche_cluster_score = trajectory_path_to_NC_score(niche_trajectory_path)
    niche_level_NTScore = niche_cluster_loading @ niche_cluster_score
    return niche_cluster_score, niche_level_NTScore


def niche_to_cell_NTScore(dataset: SpatailOmicsDataset, rel_params: Dict,
                          niche_level_NTScore: ndarray) -> Tuple[ndarray, Dict[str, ndarray], Dict[str, ndarray]]:
    """
    get cell-level NTScore
    :param dataset: SpatailOmicsDataset, dataset
    :param rel_params: Dict, relative paths
    :param niche_level_NTScore: ndarray, niche-level NTScore
    :return: Tuple[ndarray, Dict[str, ndarray], Dict[str, ndarray]], the cell-level NTScore, all niche-level NTScore dict,
    and all cell-level NTScore dict
    """

    info('Projecting NTScore from niche-level to cell-level.')

    cell_level_NTScore = np.zeros(niche_level_NTScore.shape[0])

    all_niche_level_NTScore_dict: Dict[str, ndarray] = {}
    all_cell_level_NTScore_dict: Dict[str, ndarray] = {}

    for i, data in enumerate(dataset):
        # the slice of data in each sample
        mask = data.mask
        slice_ = slice(i * data.x.shape[0], i * data.x.shape[0] + mask.sum())

        # niche to cell matrix
        niche_weight_matrix = load_npz(rel_params['Data'][i]['NicheWeightMatrix'])
        niche_to_cell_matrix = (
            niche_weight_matrix /
            niche_weight_matrix.sum(axis=0)).T  # normalize by the all niches associated with each cell, N x N

        # cell-level NTScore
        niche_level_NTScore_ = niche_level_NTScore[slice_].reshape(-1, 1)  # N x 1
        cell_level_NTScore_ = niche_to_cell_matrix @ niche_level_NTScore_
        cell_level_NTScore[slice_] = cell_level_NTScore_.reshape(-1)

        all_niche_level_NTScore_dict[data.name] = niche_level_NTScore_
        all_cell_level_NTScore_dict[data.name] = cell_level_NTScore_

    return cell_level_NTScore, all_niche_level_NTScore_dict, all_cell_level_NTScore_dict


def NTScore_table(options: Values, rel_params: Dict, all_niche_level_NTScore_dict: Dict[str, ndarray],
                  all_cell_level_NTScore_dict: Dict[str, ndarray]) -> None:
    """
    Generate NTScore table and save it
    :param options: Values, options
    :param rel_params: Dict, relative paths
    :param all_niche_level_NTScore_dict: Dict[str, ndarray], all niche-level NTScore dict
    :param all_cell_level_NTScore_dict: Dict[str, ndarray], all cell-level NTScore dict
    :return: pd.DataFrame, NTScore table
    """

    info('Output NTScore tables.')

    NTScore_table = pd.DataFrame()
    for sample in rel_params['Data']:
        coordinates_df = pd.read_csv(sample['Coordinates'], index_col=0)
        coordinates_df['Niche_NTScore'] = all_niche_level_NTScore_dict[sample['Name']]
        coordinates_df['Cell_NTScore'] = all_cell_level_NTScore_dict[sample['Name']]
        coordinates_df.to_csv(f'{options.NTScore_dir}/{sample["Name"]}_NTScore.csv.gz')
        NTScore_table = pd.concat([NTScore_table, coordinates_df])

    NTScore_table.to_csv(f'{options.NTScore_dir}/NTScore.csv.gz')
