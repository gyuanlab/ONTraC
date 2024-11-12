from optparse import Values
from typing import Callable, Dict

import numpy as np

from ..data import load_dataset
from ..GNN import evaluate, predict, save_graph_pooling_results, set_seed, train
from ..log import *
from ..model import GNN
from ..niche_net import construct_niche_network, gen_samples_yaml
from ..niche_trajectory import NTScore_table, get_niche_NTScore, load_consolidate_data, niche_to_cell_NTScore
from ..optparser import *
from ..preprocessing.pp_control import preprocessing_gnn, preprocessing_nn
from ..train import GNNBatchTrain
from ..train.inspect_funcs import get_inspect_funcs
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


def niche_network_construct(options: Values) -> None:
    """
    Niche network construct process.
    :param options: options.
    :return: None.
    """

    info('------------- Niche network construct --------------- ')

    # load input information
    meta_df, ct_coding = preprocessing_nn(meta_input=options.meta_input,
                                          NN_dir=options.NN_dir,
                                          low_res_exp_input=options.low_res_exp_input,
                                          dc_method=options.dc_method,
                                          dc_cell_type_num=options.dc_cell_type_num)

    # construct niche network
    info(message='            -------- niche network -------           ')
    construct_niche_network(meta_df=meta_df,
                            ct_coding=ct_coding,
                            save_dir=options.NN_dir,
                            n_neighbors=options.n_neighbors,
                            n_local=options.n_local)

    # generate samples.yaml to indicate file paths for each sample
    gen_samples_yaml(meta_df=meta_df, save_dir=options.NN_dir)
    info('------------ Niche network construct end ------------ ')


def gnn(options: Values) -> None:
    """
    GNN training and prediction process.
    :param options: options.
    :return: None.
    """

    info('------------------------ GNN ------------------------ ')
    # load data
    dataset, sample_loader, meta_data_df = preprocessing_gnn(NN_dir=options.NN_dir, batch_size=options.batch_size)
    # random seed
    set_seed(seed=options.seed)
    # device
    device = options.device
    # build model
    input_feats = dataset.num_features
    model = GNN(input_feats=input_feats, hidden_feats=options.hidden_feats, k=options.k, exponent=options.beta)
    # train
    loss_weight_args: Dict[str, float] = {
        key: value
        for key, value in options.__dict__.items() if key.endswith('loss_weight')
    }
    batch_train = train(nn_model=model,
                        BatchTrain=GNNBatchTrain,
                        sample_loader=sample_loader,
                        device=device,
                        max_epochs=options.epochs,
                        max_patience=options.patience,
                        min_delta=options.min_delta,
                        min_epochs=options.min_epochs,
                        lr=options.lr,
                        save_dir=options.GNN_dir,
                        inspect_funcs=get_inspect_funcs(),
                        **loss_weight_args)
    # evaluate
    evaluate(batch_train=batch_train)
    # predict
    consolidate_s_array, _ = predict(output_dir=options.GNN_dir, batch_train=batch_train, dataset=dataset)
    # save results
    if consolidate_s_array is not None:
        save_graph_pooling_results(meta_data_df=meta_data_df,
                                   dataset=dataset,
                                   rel_params=get_rel_params(NN_dir=options.NN_dir,
                                                             params=read_yaml_file(f'{options.NN_dir}/samples.yaml')),
                                   consolidate_s_array=consolidate_s_array,
                                   output_dir=options.GNN_dir)
    info('--------------------- GNN end ---------------------- ')


def niche_trajectory_construct(options: Values) -> None:
    """
    Pseudotime calculateion process.
    :param options: options.
    :return: None.
    """

    info('----------------- Niche trajectory ------------------ ')
    consolidate_s_array, consolidate_out_adj_array = load_consolidate_data(GNN_dir=options.GNN_dir)

    params = read_yaml_file(f'{options.NN_dir}/samples.yaml')
    rel_params = get_rel_params(NN_dir=options.NN_dir, params=params)
    dataset = load_dataset(NN_dir=options.NN_dir)

    niche_cluster_score, niche_level_NTScore = get_niche_NTScore(
        trajectory_construct_method=options.trajectory_construct,
        niche_cluster_loading=consolidate_s_array,
        niche_adj_matrix=consolidate_out_adj_array)
    cell_level_NTScore, all_niche_level_NTScore_dict, all_cell_level_NTScore_dict = niche_to_cell_NTScore(
        dataset=dataset, rel_params=rel_params, niche_level_NTScore=niche_level_NTScore)

    np.savetxt(fname=f'{options.NT_dir}/niche_cluster_score.csv.gz', X=niche_cluster_score, delimiter=',')
    np.savetxt(fname=f'{options.NT_dir}/niche_NTScore.csv.gz', X=niche_level_NTScore, delimiter=',')
    np.savetxt(fname=f'{options.NT_dir}/cell_NTScore.csv.gz', X=cell_level_NTScore, delimiter=',')

    NTScore_table(save_dir=options.NT_dir,
                  rel_params=rel_params,
                  all_niche_level_NTScore_dict=all_niche_level_NTScore_dict,
                  all_cell_level_NTScore_dict=all_cell_level_NTScore_dict)
    info('--------------- Niche trajectory end ---------------- ')
