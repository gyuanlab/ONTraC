from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial import cKDTree

from ..log import *


def build_knn_network(sample_name: str,
                      sample_meta_df: pd.DataFrame,
                      n_neighbors: int = 50,
                      n_local: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build KNN network for a sample.
    :param sample_name: str, sample name.
    :param sample_data_df: pd.DataFrame, sample data.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray] (coordinates, dis_matrix, indices_matrix).
    """

    info(f'Building KNN network for sample: {sample_name}...')

    # build KDTree
    coordinates = sample_meta_df[['x', 'y']].values
    kdtree = cKDTree(data=coordinates)
    dis_matrix, indices_matrix = kdtree.query(x=coordinates, k=np.max([n_neighbors, n_local]) + 1)  # include self

    return coordinates, dis_matrix, indices_matrix


def calc_edge_index(sample_name: str,
                    sample_meta_df: pd.DataFrame,
                    indices_matrix: np.ndarray,
                    n_neighbors: int = 50) -> np.ndarray:
    """
    Calculate edge index.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param indices_matrix: np.ndarray, indices matrix.
    :param n_neighbors: int, number of neighbors.
    :return: np.ndarray, edge index.
    """

    info(f'Calculating edge index for sample: {sample_name}...')

    N = sample_meta_df.shape[0]

    # save edge index file
    # 1) convert edge index to csr_matrix
    # 2) make it bidirectional
    # 3) convert it to edge index back
    # 4) save it
    src_indices = np.repeat(np.arange(sample_meta_df.shape[0]), n_neighbors)
    dst_indices = indices_matrix[:, 1:n_neighbors + 1].flatten()  # remove self, N x k, #niche x #cell
    adj_matrix = csr_matrix((np.ones(dst_indices.shape[0]), (src_indices, dst_indices)),
                            shape=(N, N))  # convert to csr_matrix
    adj_matrix = adj_matrix + adj_matrix.transpose()  # make it bidirectional
    edge_index = np.argwhere(adj_matrix > 0)  # convert it to edge index back

    return edge_index


def gauss_dist_1d(dist: np.ndarray, n_local: int) -> float:
    """
    Compute gaussian affinity between two cells (a cell and its KNN).
    :param dist: np.ndarray, distance.
    :param n_local: int, index of distance used for normalization.
    :return: float, gaussian affinity.
    """
    return np.exp(-(dist / dist[n_local])**2)


def calc_niche_weight_matrix(sample_name: str,
                             sample_meta_df: pd.DataFrame,
                             dis_matrix: np.ndarray,
                             indices_matrix: np.ndarray,
                             n_neighbors: int = 50,
                             n_local: int = 20) -> csr_matrix:
    """
    Calculate niche_weight_matrix and normalize it using self node and n_local-th neighbor using a gaussian kernel.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param dis_matrix: np.ndarray, distance matrix.
    :param indices_matrix: np.ndarray, indices matrix.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: csr_matrix, niche weight matrix.
    """

    info(f'Calculating niche weight matrix for sample: {sample_name}...')

    N = sample_meta_df.shape[0]
    niche_weight_matrix = np.apply_along_axis(func1d=gauss_dist_1d, axis=1, arr=dis_matrix,
                                              n_local=n_local)[:, :n_neighbors + 1]  # N x (k + 1), #niche x #cell
    src_indices = np.repeat(np.arange(sample_meta_df.shape[0]), n_neighbors + 1)
    dst_indices = indices_matrix[:, :n_neighbors + 1].flatten()  # include self, N x (k + 1), #niche x #cell
    niche_weight_matrix_csr = csr_matrix((niche_weight_matrix.flatten(), (src_indices, dst_indices)),
                                         shape=(N, N))  # convert to csr_matrix, N x N, #niche x #cell

    return niche_weight_matrix_csr


def calc_cell_type_composition(sample_name: str,
                               sample_meta_df: pd.DataFrame,
                               niche_weight_matrix: csr_matrix,
                               decompsited_cell_type: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate cell type composition.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param niche_weight_matrix: csr_matrix, niche weight matrix.
    :param decompsited_cell_type: Optional[np.ndarray], decompsited cell type.
    :return: np.ndarray, cell type composition.
    """

    info(f'Calculating cell type composition for sample: {sample_name}...')

    N = sample_meta_df.shape[0]

    # calculate cell type composition
    cell_to_niche_matrix = niche_weight_matrix / niche_weight_matrix.sum(axis=1)  # N x N, #niche x #cell
    if decompsited_cell_type is None:
        one_hot_matrix = np.zeros(shape=(N, sample_meta_df['Cell_Type'].cat.categories.shape[0]))  # N x #cell_type
        one_hot_matrix[np.arange(N), sample_meta_df.Cell_Type.cat.codes.values] = 1
        cell_type_composition = cell_to_niche_matrix @ one_hot_matrix  # N x n_cell_type
    else:
        cell_type_composition = cell_to_niche_matrix @ decompsited_cell_type  # N (#niche) x #cell_type

    return cell_type_composition


def save_niche_network(sample_name: str, sample_meta_df: pd.DataFrame, indices_matrix: np.ndarray,
                       edge_index: np.ndarray, niche_weight_matrix: csr_matrix, cell_type_composition: np.ndarray,
                       save_dir: Union[str, Path]) -> None:
    """
    Save the results to disk.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param indices_matrix: np.ndarray, indices matrix.
    :param edge_index: np.ndarray, edge index.
    :param niche_weight_matrix: csr_matrix, niche weight matrix.
    :param cell_type_composition: np.ndarray, cell type composition.
    :param save_dir: str, save directory.
    :return: None.
    """

    # save coordinates
    # TODO: support 3D coordinates
    if 'Cell_ID' in sample_meta_df.columns:
        coord_df = sample_meta_df[['Cell_ID', 'x', 'y']]
    elif 'Spot_ID' in sample_meta_df.columns:
        coord_df = sample_meta_df[['Spot_ID', 'x', 'y']]
    else:
        raise ValueError('Cell_ID or Spot_ID is required for saving coordinates.')
    coord_df.to_csv(f'{save_dir}/{sample_name}_Coordinates.csv', index=False)

    # save the kNN network
    neighbor_indices_file = f'{save_dir}/{sample_name}_NeighborIndicesMatrix.csv.gz'
    np.savetxt(fname=neighbor_indices_file, X=indices_matrix, delimiter=',')  # save indices matrix
    edge_index_file = f'{save_dir}/{sample_name}_EdgeIndex.csv.gz'
    np.savetxt(edge_index_file, edge_index, delimiter=',', fmt='%d')  # save edge index

    # save niche_weight_matrix
    niche_weight_file = f'{save_dir}/{sample_name}_NicheWeightMatrix.npz'
    save_npz(file=niche_weight_file, matrix=niche_weight_matrix)  # save weight matrix

    # save cell type composition
    cell_type_composition_file = f'{save_dir}/{sample_name}_CellTypeComposition.csv.gz'
    np.savetxt(fname=cell_type_composition_file, X=cell_type_composition, delimiter=',')


def construct_niche_network_sample(sample_name: str,
                                   sample_meta_df: pd.DataFrame,
                                   sample_ct_coding: pd.DataFrame,
                                   save_dir: Union[str, Path],
                                   n_neighbors: int = 50,
                                   n_local: int = 20) -> None:
    """
    Construct niche network for a sample.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param sample_ct_coding: pd.DataFrame, sample cell type coding.
    :param save_dir: str, save directory.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: None.

    1) get coordinates and save it.
    2) save the celltype information.
    3) build KDTree.
        1. save edge index file.
        2. calculate weight matrix.
        3. calculate cell type composition and save it.
    """

    info(f'Constructing niche network for sample: {sample_name}.')

    # build kNN network
    coordinates, dis_matrix, indices_matrix = build_knn_network(sample_name=sample_name,
                                                                sample_meta_df=sample_meta_df,
                                                                n_neighbors=n_neighbors,
                                                                n_local=n_local)

    # calculate edge index
    edge_index = calc_edge_index(sample_name=sample_name,
                                 sample_meta_df=sample_meta_df,
                                 indices_matrix=indices_matrix,
                                 n_neighbors=n_neighbors)

    # calculate niche_weight_matrix
    niche_weight_matrix = calc_niche_weight_matrix(sample_name=sample_name,
                                                   sample_meta_df=sample_meta_df,
                                                   dis_matrix=dis_matrix,
                                                   indices_matrix=indices_matrix,
                                                   n_neighbors=n_neighbors,
                                                   n_local=n_local)

    # calculate cell type composition
    cell_type_composition = calc_cell_type_composition(sample_meta_df=sample_meta_df,
                                                       niche_weight_matrix=niche_weight_matrix,
                                                       sample_name=sample_name,
                                                       decompsited_cell_type=sample_ct_coding.values)

    save_niche_network(sample_meta_df=sample_meta_df,
                       sample_name=sample_name,
                       indices_matrix=indices_matrix,
                       edge_index=edge_index,
                       niche_weight_matrix=niche_weight_matrix,
                       cell_type_composition=cell_type_composition,
                       save_dir=save_dir)


def construct_niche_network(meta_df: pd.DataFrame,
                            ct_coding: pd.DataFrame,
                            save_dir: Union[str, Path],
                            n_neighbors: int = 50,
                            n_local: int = 20) -> None:
    """
    Construct niche network.
    :param meta_df: pd.DataFrame, meta data.
    :param ct_coding: pd.DataFrame, cell type coding.
    :param save_dir: str, save directory.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: None.
    """

    # get samples
    samples = meta_df['Sample'].unique()

    # construct niche network for each sample
    for sample_name in samples:
        sample_spot_ids = meta_df[meta_df['Sample'] == sample_name]['Spot_ID']
        sample_meta_df = meta_df[meta_df['Sample'] == sample_name]
        sample_ct_coding = ct_coding.loc[sample_spot_ids]
        construct_niche_network_sample(sample_name=sample_name,
                                       sample_meta_df=sample_meta_df,
                                       sample_ct_coding=sample_ct_coding,
                                       save_dir=save_dir,
                                       n_neighbors=n_neighbors,
                                       n_local=n_local)


def gen_samples_yaml(meta_df: pd.DataFrame, save_dir: Union[str, Path]) -> None:
    """
    Generate samples.yaml.
    :param meta_df: pd.DataFrame, meta data.
    :param save_dir: str, save directory.
    :return: None.
    """

    info('Generating samples.yaml file.')

    data: Dict[str, List[Any]] = {'Data': []}
    for sample in meta_df['Sample'].unique():
        data['Data'].append({
            'Name': f'{sample}',
            'Coordinates': f'{sample}_Coordinates.csv',
            'EdgeIndex': f'{sample}_EdgeIndex.csv.gz',
            'Features': f'{sample}_CellTypeComposition.csv.gz',
            'NicheWeightMatrix': f'{sample}_NicheWeightMatrix.npz',
            'NeighborIndicesMatrix': f'{sample}_NeighborIndicesMatrix.csv.gz'
        })

    yaml_file = f'{save_dir}/samples.yaml'
    with open(yaml_file, 'w') as fhd:
        yaml.dump(data, fhd)
