from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch_geometric.loader import DenseDataLoader

from ..data import SpatailOmicsDataset, load_dataset
from ..external.deconvolution import apply_STdeconvolve
from ..log import info, warning
from ..utils import get_meta_data_file
from .data import load_meta_data, save_cell_type_code


def load_input_data(meta_input: Union[str, Path],
                    NN_dir: Union[str, Path],
                    low_res_exp_input: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data from original inputs.
    :param meta_input: str or Path, meta data file.
    :param NN_dir: str or Path, save directory.
    :param low_res_exp_input: str or Path, low resolution expression file. Gene X Spot matrix in csv format.
    :return: Dict[str, pd.DataFrame], loaded data.
    """

    output = {}

    # load meta_data
    meta_data_df = load_meta_data(save_dir=NN_dir, meta_data_file=meta_input)
    output['meta_data'] = meta_data_df

    # load low resolution expression data
    if low_res_exp_input is not None:
        low_res_exp_df = pd.read_csv(low_res_exp_input, header=0, index_col=0, sep=',')

        # validate
        if 'Spot_ID' not in meta_data_df.columns:
            raise ValueError('Spot_ID in metadata input is required for spot-level data.')
        if not set(meta_data_df['Spot_ID']).issubset(low_res_exp_df.columns.to_list()):
            raise ValueError('There are spots in metadata that are not in low resolution expression data.')
        if not set(low_res_exp_df.columns.to_list()).issubset(meta_data_df['Spot_ID']):
            warning(
                'There are spots in low resolution expression data that are not in metadata. These spots will be ignored.'
            )
            low_res_exp_df = low_res_exp_df[meta_data_df['Spot_ID'].to_list()]

        # save
        output['low_res_exp'] = low_res_exp_df

    else:  # cell-level data, validate the meta_data
        if 'Cell_Type' not in meta_data_df.columns:
            raise ValueError('Cell_Type in metadata is required for cell-level data.')
        if 'Cell_ID' not in meta_data_df.columns:
            raise ValueError('Cell_ID in metadata is required for cell-level data.')

    return output


def perform_deconvolution(NN_dir: Union[str, Path], dc_method: str, exp_matrix: np.ndarray,
                          dc_cell_type_num: int) -> np.ndarray:
    """
    Perform deconvolution.
    :param NN_dir: str or Path, save directory.
    :param dc_method: str, deconvolution method.
    :param exp_matrix: np.ndarray, expression matrix.
    :param dc_cell_type_num: int, number of cell types.
    :return: np.ndarray, deconvoluted cell type matrix.
    """

    if dc_method == 'STdeconvolve':
        deconvoluted_ct_matrix = apply_STdeconvolve(NN_dir=NN_dir, exp_matrix=exp_matrix, ct_num=dc_cell_type_num)

    return deconvoluted_ct_matrix


def preprocessing_nn(meta_input: Union[str, Path],
                     NN_dir: Union[str, Path],
                     low_res_exp_input: Optional[Union[str, Path]] = None,
                     dc_method: Optional[str] = None,
                     dc_cell_type_num: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocessing for niche network.
    :param meta_input: str or Path, meta data file.
    :param NN_dir: str or Path, save directory.
    :param low_res_exp_input: str or Path, low resolution expression file.
    :param dc_method: str, deconvolution method.
    :param dc_cell_type_num: int, number of cell types.
    :return: Tuple[pd.DataFrame, pd.DataFrame], meta data and cell type coding matrix.
    """

    input_data = load_input_data(meta_input=meta_input, NN_dir=NN_dir, low_res_exp_input=low_res_exp_input)

    if 'meta_data' not in input_data:
        raise ValueError('meta_data is required for preprocessing.')
    meta_df = input_data['meta_data']

    # cell or spot level data
    if 'low_res_exp' not in input_data:  # cell-level data
        # generate cell type coding matrix
        meta_df['Cell_Type'] = meta_df['Cell_Type'].astype('category')
        ct_coding_matrix = np.zeros(shape=(meta_df.shape[0],
                                           meta_df['Cell_Type'].cat.categories.shape[0]))  # N x #cell_type
        ct_coding_matrix[np.arange(meta_df.shape[0]), meta_df.Cell_Type.cat.codes.values] = 1
        ct_coding = pd.DataFrame(data=ct_coding_matrix,
                                 columns=meta_df['Cell_Type'].cat.categories,
                                 index=meta_df.index)

        # save cell type code
        save_cell_type_code(save_dir=NN_dir, cell_types=meta_df['Cell_Type'])

    else:  # spot-level data
        if dc_method is None or dc_cell_type_num is None:
            raise ValueError('dc_method and dc_cell_type_num are required for spot-level data.')
        ct_coding_matrix = perform_deconvolution(NN_dir=NN_dir,
                                                 dc_method=dc_method,
                                                 exp_matrix=input_data['low_res_exp'],
                                                 dc_cell_type_num=dc_cell_type_num)
        ct_coding = pd.DataFrame(data=ct_coding_matrix,
                                 columns=np.arange(ct_coding_matrix.shape[1]),
                                 index=input_data['low_res_exp'].index)
        ct_coding.to_csv(f'{NN_dir}/spotxcelltype.csv.gz', index=True)

        # save cell type code
        save_cell_type_code(save_dir=NN_dir, cell_types=pd.Series(ct_coding.columns))

    return meta_df, ct_coding


def load_data(NN_dir: Union[str, Path], batch_size: int = 0) -> Tuple[SpatailOmicsDataset, DenseDataLoader]:
    """
    Load data and create sample loader.
    :param NN_dir: str or Path, save directory.
    :param batch_size: int, batch size.
    :return: Tuple[SpatailOmicsDataset, DenseDataLoader], dataset and sample loader.
    """

    info('Loading dataset.')

    dataset = load_dataset(NN_dir=NN_dir)
    batch_size = batch_size if batch_size > 0 else len(dataset)
    sample_loader = DenseDataLoader(dataset, batch_size=batch_size)

    return dataset, sample_loader


def preprocessing_gnn(NN_dir: Union[str, Path],
                      batch_size: int = 0) -> Tuple[SpatailOmicsDataset, DenseDataLoader, pd.DataFrame]:
    """
    Preprocessing for GNN.
    :param NN_dir: str or Path, save directory.
    :param batch_size: int, batch size.
    :return: Tuple[SpatailOmicsDataset, DenseDataLoader, pd.DataFrame], dataset, sample loader, and meta data.
    """

    # meta data
    meta_data_df = pd.read_csv(get_meta_data_file(NN_dir), header=0)

    # dataset and sample loader
    dataset, sample_loader = load_data(NN_dir=NN_dir, batch_size=batch_size)

    return dataset, sample_loader, meta_data_df
