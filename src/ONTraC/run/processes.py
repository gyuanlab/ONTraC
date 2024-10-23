from optparse import Values
from typing import Callable, List, Optional, Type

import numpy as np
import pandas as pd
import torch

from ..data import load_dataset
from ..GNN import (evaluate, load_data, predict, save_graph_pooling_results,
                   set_seed, train)
from ..log import *
from ..niche_net import construct_niche_network, gen_samples_yaml
from ..niche_trajectory import (NTScore_table, get_niche_NTScore,
                                load_consolidate_data, niche_to_cell_NTScore)
from ..train import SubBatchTrainProtocol
from ..utils import get_rel_params, read_yaml_file


def load_parameters(opt_validate_func: Callable, prepare_optparser_func: Callable) -> Values:
    """
    Load parameters.
    :param opt_validate_func: validate function.
    :param prepare_optparser_func: prepare optparser function.
    :return: options.
    """
    options = opt_validate_func(prepare_optparser_func())

    return options


def niche_network_construct(options: Values, ori_data_df: pd.DataFrame) -> None:
    """
    Niche network construct process.
    :param options: options.
    :param ori_data_df: pd.DataFrame, original data.
    :return: None.
    """

    info('------------- Niche network construct --------------- ')
    # construct niche network
    construct_niche_network(options=options, ori_data_df=ori_data_df)

    # generate samples.yaml to indicate file paths for each sample
    gen_samples_yaml(options=options, ori_data_df=ori_data_df)
    info('------------ Niche network construct end ------------ ')


def gnn(options: Values, ori_data_df: pd.DataFrame, nn_model: Type[torch.nn.Module],
        BatchTrain: Type[SubBatchTrainProtocol], inspect_funcs: Optional[List[Callable]]) -> None:
    """
    GNN training and prediction process.
    :param options: options.
    :param ori_data_df: pd.DataFrame, original data.
    :param nn_model: nn model.
    :param BatchTrain: batch train.
    :param inspect_funcs: inspect functions.
    :return: None.
    """

    info('------------------------ GNN ------------------------ ')
    # load data
    dataset, sample_loader = load_data(options=options)
    # random seed
    set_seed(seed=options.seed)
    # build model
    input_feats = dataset.num_features
    model = nn_model(input_feats=input_feats, hidden_feats=options.hidden_feats, k=options.k, exponent=options.beta)
    # train
    batch_train = train(options=options,
                        nn_model=model,
                        BatchTrain=BatchTrain,
                        sample_loader=sample_loader,
                        inspect_funcs=inspect_funcs)
    # evaluate
    evaluate(batch_train=batch_train, model_name='GraphPooling')
    # predict
    consolidate_s_array, _ = predict(options=options,
                                     output_dir=options.GNN_dir,
                                     batch_train=batch_train,
                                     dataset=dataset,
                                     model_name='GraphPooling')
    # save results
    if consolidate_s_array is not None:
        save_graph_pooling_results(ori_data_df=ori_data_df,
                                   dataset=dataset,
                                   rel_params=get_rel_params(
                                       options=options,
                                       params=read_yaml_file(f'{options.preprocessing_dir}/samples.yaml')),
                                   consolidate_s_array=consolidate_s_array,
                                   output_dir=options.GNN_dir)
    info('--------------------- GNN end ---------------------- ')


def NTScore(options: Values) -> None:
    """
    Pseudotime calculateion process.
    :param options: options.
    :return: None.
    """

    info('----------------- Niche trajectory ------------------ ')
    consolidate_s_array, consolidate_out_adj_array = load_consolidate_data(options=options)

    params = read_yaml_file(f'{options.preprocessing_dir}/samples.yaml')
    rel_params = get_rel_params(options, params)
    dataset = load_dataset(options=options)

    niche_cluster_score, niche_level_NTScore = get_niche_NTScore(options=options,
                                                                 niche_cluster_loading=consolidate_s_array,
                                                                 niche_adj_matrix=consolidate_out_adj_array)
    cell_level_NTScore, all_niche_level_NTScore_dict, all_cell_level_NTScore_dict = niche_to_cell_NTScore(
        dataset=dataset, rel_params=rel_params, niche_level_NTScore=niche_level_NTScore)

    np.savetxt(fname=f'{options.NTScore_dir}/niche_cluster_score.csv.gz', X=niche_cluster_score, delimiter=',')
    np.savetxt(fname=f'{options.NTScore_dir}/niche_NTScore.csv.gz', X=niche_level_NTScore, delimiter=',')
    np.savetxt(fname=f'{options.NTScore_dir}/cell_NTScore.csv.gz', X=cell_level_NTScore, delimiter=',')

    NTScore_table(options=options,
                  rel_params=rel_params,
                  all_niche_level_NTScore_dict=all_niche_level_NTScore_dict,
                  all_cell_level_NTScore_dict=all_cell_level_NTScore_dict)
    info('--------------- Niche trajectory end ---------------- ')
