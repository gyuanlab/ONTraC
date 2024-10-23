import sys
from optparse import OptionGroup, OptionParser, Values
from random import randint

import torch

from ..log import *


def add_train_options_group(optparser: OptionParser) -> OptionGroup:
    """
    Add train options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """
    # overall train options group
    group_train = OptionGroup(optparser, "Options for GNN training")
    optparser.add_option_group(group_train)
    group_train.add_option('--device',
                           dest='device',
                           type='string',
                           help='Device for training. We support cpu and cuda now. Auto select if not specified.')
    group_train.add_option('--epochs',
                           dest='epochs',
                           type='int',
                           default=1000,
                           help='Number of maximum epochs for training. Default is 1000.')
    group_train.add_option('--patience',
                           dest='patience',
                           type='int',
                           default=100,
                           help='Number of epochs wait for better result. Default is 100.')
    group_train.add_option('--min-delta',
                           dest='min_delta',
                           type='float',
                           default=0.001,
                           help='Minimum delta for better result. Default is 0.001')
    group_train.add_option('--min-epochs',
                           dest='min_epochs',
                           type='int',
                           default=50,
                           help='Minimum number of epochs for training. Default is 50. Set to 0 to disable.')
    group_train.add_option('--batch-size',
                           dest='batch_size',
                           type='int',
                           default=0,
                           help='Batch size for training. Default is 0 for whole dataset.')
    group_train.add_option('-s', '--seed', dest='seed', type='int', help='Random seed for training. Default is random.')
    group_train.add_option('--lr',
                           dest='lr',
                           type='float',
                           default=0.03,
                           help='Learning rate for training. Default is 0.03.')
    return group_train


def add_GNN_options_group(group_train: OptionGroup) -> None:
    """
    Add GNN options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # GNN options group
    group_train.add_option('--hidden-feats',
                           dest='hidden_feats',
                           type='int',
                           default=4,
                           help='Number of hidden features. Default is 4.')
    group_train.add_option('--degree-out-adj-norm',
                           dest='degree_out_adj_norm',
                           action='store_true',
                           default=False,
                           help='Perform degree normalization on adjacency matrix.')
    group_train.add_option('--expectation-out-adj-norm',
                           dest='expectation_out_adj_norm',
                           action='store_true',
                           default=False,
                           help='Perform expected random assignment normalization on adjacency matrix.')


def add_NP_options_group(group_train: OptionGroup) -> None:
    """
    Add Node Pooling options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # NP options group
    group_train.add_option('-k',
                           '--k-clusters',
                           dest='k',
                           type='int',
                           default=6,
                           help='Number of niche clusters. Default is 6.')
    group_train.add_option('--modularity-loss-weight',
                           dest='modularity_loss_weight',
                           type='float',
                           default=0.3,
                           help='Weight for modularity loss. Default is 0.3.')
    group_train.add_option('--purity-loss-weight',
                           dest='purity_loss_weight',
                           type='float',
                           default=300,
                           help='Weight for purity loss. Default is 300.')
    group_train.add_option('--regularization-loss-weight',
                           dest='regularization_loss_weight',
                           type='float',
                           default=0.1,
                           help='Weight for regularization loss. Default is 0.1.')
    group_train.add_option('--beta',
                           dest='beta',
                           type='float',
                           default=0.03,
                           help='Beta value control niche cluster assignment matrix. Default is 0.03.')


def validate_train_options(optparser: OptionParser, options: Values) -> Values:
    """
    Validate train options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    # device
    if options.device is None:
        info('Device not specified, choose automatically.')
    elif options.device.startswith(('cuda', 'cpu')):
        if options.device.startswith('cuda') and not torch.cuda.is_available():
            warning('CUDA is not available, use CPU instead.')
            options.device = 'cpu'
    else:
        warning(f'Invalid device {options.device}! Choose automatically.')
        options.device = None
    if options.device is None:
        if torch.cuda.is_available():
            options.device = 'cuda'
        # elif torch.backends.mps.is_available():  # TODO: MPS compatibility with torch_geometric.data.InMemoryDataset
        #     options.device = 'mps'
        else:
            options.device = 'cpu'
    
    # determin random seed
    if getattr(options, 'seed') is None:
        options.seed = randint(0, 10000)

    return options


def validate_GNN_options(optparser: OptionParser, options: Values) -> Values:
    """
    Validate GNN options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    # check hidden_feats
    if getattr(options, 'hidden_feats') < 2:
        error(f'hidden_feats must be greater than 1, exit!')
        sys.exit(1)

    return options


def validate_NP_options(optparser: OptionParser, options: Values) -> Values:
    """
    Validate Node Pooling options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    # check k
    if getattr(options, 'k') < 2:
        error(f'k must be greater than 1, exit!')
        sys.exit(1)

    return options


def write_train_options_memo(options: Values) -> None:
    """
    Write train options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info('           -------- train options -------            ')
    info(f'device:  {options.device}')
    info(f'epochs:  {options.epochs}')
    info(f'batch_size:  {options.batch_size}')
    info(f'patience:  {options.patience}')
    info(f'min_delta:  {options.min_delta}')
    info(f'min_epochs:  {options.min_epochs}')
    info(f'seed:  {options.seed}')
    info(f'lr:  {options.lr}')


def write_GNN_options_memo(options: Values) -> None:
    """
    Write GNN options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'hidden_feats:  {options.hidden_feats}')
    if hasattr(options, 'degree_out_adj_norm'):
        info(f'Degree normalization on adjacency matrix: {options.degree_out_adj_norm}')
    if hasattr(options, 'expectation_out_adj_norm'):
        info(f'Expected random assignment normalization on adjacency matrix: {options.expectation_out_adj_norm}')


def write_NP_options_memo(options: Values) -> None:
    """
    Write Node Pooling options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'k:  {options.k}')
    info(f'modularity_loss_weight:  {options.modularity_loss_weight}')
    info(f'purity_loss_weight:  {options.purity_loss_weight}')
    info(f'regularization_loss_weight:  {options.regularization_loss_weight}')
    info(f'beta:  {options.beta}')


__all__ = [
    'add_train_options_group',
    'add_GNN_options_group',
    'add_NP_options_group',
    'validate_train_options',
    'validate_GNN_options',
    'validate_NP_options',
    'write_train_options_memo',
    'write_GNN_options_memo',
    'write_NP_options_memo',
]
