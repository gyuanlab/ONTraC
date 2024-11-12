import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from torch import Tensor
from torch_geometric.loader import DenseDataLoader

from ..data import SpatailOmicsDataset
from ..log import info
from ..train import SubBatchTrainProtocol


def set_seed(seed: int) -> None:
    """
    Set seed.
    :param seed: seed.
    :return: None.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(nn_model: torch.nn.Module,
          BatchTrain: Type[SubBatchTrainProtocol],
          sample_loader: DenseDataLoader,
          device: torch.device,
          max_epochs: int,
          max_patience: int,
          min_delta: float,
          min_epochs: int,
          lr: float,
          save_dir: Union[str, Path],
          inspect_funcs: Optional[List[Callable]] = None,
          **kwargs) -> SubBatchTrainProtocol:
    """
    GNN training process.
    :param nn_model: nn model.
    :param BatchTrain: Type[SubBatchTrainProtocol], batch train.
    :param sample_loader: DenseDataLoader, sample loader.
    :param device: torch.device, device.
    :param max_epochs: int, max epochs.
    :param max_patience: int, max patience.
    :param min_delta: float, min delta.
    :param min_epochs: int, min epochs.
    :param lr: float, learning rate.
    :param save_dir: Union[str, Path], save directory.
    :param inspect_funcs: Optional[List[Callable]], inspect functions.
    :param kwargs: dict, loss weight arguments.
    """
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    batch_train = BatchTrain(model=nn_model, device=torch.device(device), data_loader=sample_loader)  # type: ignore
    batch_train.save(path=f'{save_dir}/epoch_0.pt')

    loss_weight_args: Dict[str, float] = {key: value for key, value in kwargs.items() if key.endswith('loss_weight')}

    batch_train.train(optimizer=optimizer,
                      inspect_funcs=inspect_funcs,
                      max_epochs=max_epochs,
                      max_patience=max_patience,
                      min_delta=min_delta,
                      min_epochs=min_epochs,
                      output=save_dir,
                      **loss_weight_args)
    batch_train.save(path=f'{save_dir}/model_state_dict.pt')
    info(message=f'Training process end.')
    return batch_train


def evaluate(batch_train: SubBatchTrainProtocol) -> None:
    """
    Evaluate the performance of ONTraC model on data.
    :param batch_train: SubBatchTrainProtocol, batch train.
    :return
    """
    info(message=f'Evaluating process start.')
    loss_dict: Dict[str, np.floating] = batch_train.evaluate()  # type: ignore
    info(message=f'Evaluate loss, {repr(loss_dict)}')
    info(message=f'Evaluating process end.')


def predict(output_dir: str, batch_train: SubBatchTrainProtocol,
            dataset: SpatailOmicsDataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Predict the results of ONTraC model on data.
    :param output_dir: str, output directory.
    :param batch_train: SubBatchTrainProtocol, batch train.
    :param dataset: SpatailOmicsDataset, dataset.
    :return: consolidate_s_array, consolidate_out_adj_array.
    """
    info(f'Predicting process start.')
    each_sample_loader = DenseDataLoader(dataset, batch_size=1)
    consolidate_flag = False
    consolidate_s_list = []
    consolidate_out = None
    consolidate_out_adj = None
    for data in each_sample_loader:  # type: ignore
        info(f'Generating prediction results for {data.name[0]}.')
        data = data.to(batch_train.device)  # type: ignore
        predict_result = batch_train.predict_dict(data=data)  # type: ignore
        for key, value in predict_result.items():
            np.savetxt(fname=f'{output_dir}/{data.name[0]}_{key}.csv.gz',
                       X=value.squeeze(0).detach().cpu().numpy(),
                       delimiter=',')

        # consolidate results
        if not consolidate_flag and ('s' in predict_result and 'out' in predict_result and 'out_adj' in predict_result):
            consolidate_flag = True
        if consolidate_flag:
            s = predict_result['s']
            out = predict_result['out']
            s = s.squeeze(0)
            consolidate_s_list.append(s)
            out_adj_ = torch.matmul(torch.matmul(s.T, data.adj.squeeze(0)), s)
            consolidate_out_adj: Tensor = out_adj_ if consolidate_out_adj is None else consolidate_out_adj + out_adj_
            consolidate_out: Tensor = out.squeeze(
                0) * data.mask.sum() if consolidate_out is None else consolidate_out + out.squeeze(0) * data.mask.sum()

    consolidate_s_array, consolidate_out_adj_array = None, None
    if consolidate_flag:
        # consolidate out
        nodes_num = 0
        for data in each_sample_loader:  # type: ignore
            nodes_num += data.mask.sum()
        consolidate_out = consolidate_out / nodes_num  # type: ignore
        consolidate_out_array = consolidate_out.detach().cpu().numpy()  # type: ignore
        np.savetxt(fname=f'{output_dir}/consolidate_out.csv.gz', X=consolidate_out_array, delimiter=',')
        # consolidate s
        consolidate_s = torch.cat(consolidate_s_list, dim=0)
        # consolidate out_adj
        ind = torch.arange(consolidate_s.shape[-1], device=consolidate_out_adj.device)  # type: ignore
        consolidate_out_adj[ind, ind] = 0  # type: ignore
        d = torch.einsum('ij->i', consolidate_out_adj)
        d = torch.sqrt(d)[:, None] + 1e-15
        consolidate_out_adj = (consolidate_out_adj / d) / d.transpose(0, 1)
        consolidate_s_array = consolidate_s.detach().cpu().numpy()
        consolidate_out_adj_array = consolidate_out_adj.detach().cpu().numpy()
        np.savetxt(fname=f'{output_dir}/consolidate_s.csv.gz', X=consolidate_s_array, delimiter=',')
        np.savetxt(fname=f'{output_dir}/consolidate_out_adj.csv.gz', X=consolidate_out_adj_array, delimiter=',')

    info(f'Predicting process end.')
    return consolidate_s_array, consolidate_out_adj_array


def save_graph_pooling_results(meta_data_df: pd.DataFrame, dataset: SpatailOmicsDataset, rel_params: Dict,
                               consolidate_s_array: np.ndarray, output_dir: str) -> None:
    """
    Save graph pooling results as the Niche cluster (max probability for each niche & cell).
    :param meta_data_df: pd.DataFrame, original data. Sample and Cell_ID/Spot_ID columns are used.
    :param dataset: SpatailOmicsDataset, dataset.
    :param rel_params: dict, relative parameters.
    :param consolidate_s_array: np.ndarray, consolidate s array.
    :param output_dir: str, output directory.
    :return: None.
    """

    if 'Cell_ID' in meta_data_df.columns:
        id_name = 'Cell_ID'
    elif 'Spot_ID' in meta_data_df.columns:
        id_name = 'Spot_ID'
    else:
        raise ValueError('Cell_ID or Spot_ID is required in meta data.')

    consolidate_s_niche_df = pd.DataFrame()
    consolidate_s_cell_df = pd.DataFrame()
    for i, data in enumerate(dataset):
        # the slice of data in each sample
        slice_ = slice(i * data.x.shape[0], i * data.x.shape[0] + data.mask.sum())
        consolidate_s = consolidate_s_array[slice_]  # N x C
        consolidate_s_df_ = pd.DataFrame(consolidate_s,
                                         columns=[f'NicheCluster_{i}' for i in range(consolidate_s.shape[1])])
        consolidate_s_df_[id_name] = meta_data_df[meta_data_df['Sample'] == data.name][id_name].values
        consolidate_s_niche_df = pd.concat([consolidate_s_niche_df, consolidate_s_df_], axis=0)

        # niche to cell matrix
        niche_weight_matrix = load_npz(rel_params['Data'][i]['NicheWeightMatrix'])
        niche_to_cell_matrix = (
            niche_weight_matrix /
            niche_weight_matrix.sum(axis=0)).T  # normalize by the all niches associated with each cell, N x N

        consolidate_s_cell = niche_to_cell_matrix @ consolidate_s
        consolidate_s_cell_df_ = pd.DataFrame(consolidate_s_cell,
                                              columns=[f'NicheCluster_{i}' for i in range(consolidate_s_cell.shape[1])])
        consolidate_s_cell_df_[id_name] = meta_data_df[meta_data_df['Sample'] == data.name][id_name].values
        consolidate_s_cell_df = pd.concat([consolidate_s_cell_df, consolidate_s_cell_df_], axis=0)

    consolidate_s_niche_df = consolidate_s_niche_df.set_index(id_name)
    consolidate_s_niche_df = consolidate_s_niche_df.loc[meta_data_df[id_name], :]
    consolidate_s_niche_df.to_csv(f'{output_dir}/niche_level_niche_cluster.csv.gz',
                                  index=True,
                                  index_label=id_name,
                                  header=True)
    consolidate_s_niche_df['Niche_Cluster'] = consolidate_s_niche_df.values.argmax(axis=1)
    consolidate_s_niche_df['Niche_Cluster'].to_csv(f'{output_dir}/niche_level_max_niche_cluster.csv.gz',
                                                   index=True,
                                                   header=True)
    consolidate_s_cell_df = consolidate_s_cell_df.set_index(id_name)
    consolidate_s_cell_df = consolidate_s_cell_df.loc[meta_data_df[id_name], :]
    consolidate_s_cell_df.to_csv(f'{output_dir}/cell_level_niche_cluster.csv.gz',
                                 index=True,
                                 index_label=id_name,
                                 header=True)
    consolidate_s_cell_df['Niche_Cluster'] = consolidate_s_cell_df.values.argmax(axis=1)
    consolidate_s_cell_df['Niche_Cluster'].to_csv(f'{output_dir}/cell_level_max_niche_cluster.csv.gz',
                                                  index=True,
                                                  header=True)
