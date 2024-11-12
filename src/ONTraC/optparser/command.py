# niche network construction options
from optparse import OptionGroup, OptionParser, Values
from typing import List

from ..constants import IO_OPTIONS
from ..log import *
from ..optparser._analysis import *
from ..version import __version__
from ._IO import *
from ._NN import *
from ._NT import *
from ._preprocessing import *
from ._train import *


# ------------------------------------
# ONTraC functions
# ------------------------------------
def prepare_ontrac_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in this function first.
    :return: OptionParser object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NT-dir NTSCORE_DIR> <--meta-input META_INPUT> [--low-res-exp-input LOW_RES_EXP_INPUT]
    [--deconvolution-method DC_METHOD] [--deconvolution-cell-type-number DC_CELL_TYPE_NUM] [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL]
    [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED]
    [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K_CLUSTERS] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT]
    [--purity-loss-weight PURITY_LOSS_WEIGHT] [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]
    [--trajectory-construct TRAJECTORY_CONSTRUCT]'''
    description = 'All steps of ONTraC including dataset creation, Graph Pooling, and NT score calculation.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # prepprocessing options group
    add_preprocessing_options_group(optparser=optparser)

    # Niche net construction options group
    add_niche_net_constr_options_group(optparser=optparser)

    # train and model options
    group_train: OptionGroup = add_train_options_group(optparser=optparser)
    add_GCN_options_group(group_train=group_train)
    add_GP_options_group(group_train=group_train)

    # Niche trajectory options group
    add_NT_options_group(optparser=optparser)

    return optparser


def opt_ontrac_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC']  # type: ignore

    (options, args) = optparser.parse_args()

    # IO
    validate_io_options(optparser=optparser, options=options, io_options=io_options)
    # preprocessing
    validate_preprocessing_options(optparser=optparser, options=options)
    # niche network construction
    validate_niche_net_constr_options(optparser=optparser, options=options)
    # training
    validate_train_options(optparser=optparser, options=options)
    validate_GP_options(optparser=optparser, options=options)
    validate_GCN_options(optparser=optparser, options=options)
    # niche trajectory
    validate_NT_options(optparser=optparser, options=options)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_preprocessing_memo(options=options)
    write_niche_net_constr_memo(options=options)
    write_train_options_memo(options=options)
    write_GCN_options_memo(options=options)
    write_GP_options_memo(options=options)
    write_NT_options_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


# ------------------------------------
# ONTraC_NN functions
# ------------------------------------
def prepare_nn_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_NN']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir PREPROCESSING_DIR> <--meta-input META_INPUT> [--low-res-exp-input LOW_RES_EXP_INPUT]
    [--deconvolution-method DC_METHOD] [--deconvolution-cell-type-number DC_CELL_TYPE_NUM] [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL]'''
    description = 'Preporcessing and create dataset for GNN and following analysis.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # prepprocessing options group
    add_preprocessing_options_group(optparser=optparser)

    # niche network construction options group
    add_niche_net_constr_options_group(optparser=optparser)

    return optparser


def opt_nn_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_NN']  # type: ignore

    (options, args) = optparser.parse_args()

    validate_io_options(optparser=optparser, options=options, io_options=io_options)
    validate_preprocessing_options(optparser=optparser, options=options)
    validate_niche_net_constr_options(optparser=optparser, options=options)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_preprocessing_memo(options=options)
    write_niche_net_constr_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


# ------------------------------------
# ONTraC_GNN functions
# ------------------------------------
def prepare_gnn_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_GNN']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]'''
    description = 'Graph Neural Network (GNN). The core algorithm of ONTraC.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # train and model options
    group_train: OptionGroup = add_train_options_group(optparser=optparser)
    add_GCN_options_group(group_train=group_train)
    add_GP_options_group(group_train=group_train)

    return optparser


def opt_gnn_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_GNN']  # type: ignore

    (options, args) = optparser.parse_args()

    validate_io_options(optparser=optparser, options=options, io_options=io_options)
    validate_train_options(optparser=optparser, options=options)
    validate_GCN_options(optparser=optparser, options=options)
    validate_GP_options(optparser=optparser, options=options)

    info(message='------------------ RUN params memo ------------------ ')
    # print parameters to stdout
    write_io_options_memo(options=options, io_options=io_options)
    write_train_options_memo(options=options)
    write_GCN_options_memo(options=options)
    write_GP_options_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


# ------------------------------------
# ONTraC_NT functions
# ------------------------------------
def prepare_nt_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_NT']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NT-dir NTSCORE_DIR> 
            [--trajectory-construct TRAJECTORY_CONSTRUCT]'''
    description = 'ONTraC_NT: construct niche trajectory for niche cluster and project the NT score to each cell'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=io_options)
    add_NT_options_group(optparser=optparser)

    return optparser


def opt_nt_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_NT']  # type: ignore

    (options, args) = optparser.parse_args()

    validate_io_options(optparser=optparser, options=options, io_options=io_options)
    validate_NT_options(optparser=optparser, options=options)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_NT_options_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


# ------------------------------------
# ONTraC_GT functions
# ------------------------------------
def prepare_gt_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_GT']  # type: ignore

    usage = f'''USAGE: %prog <--NN-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NT-dir NTSCORE_DIR>
    [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--trajectory-construct TRAJECTORY_CONSTRUCT]'''
    description = 'ONTraC_GT: GNN and Niche Trajectory'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # train and model options
    group_train: OptionGroup = add_train_options_group(optparser=optparser)
    add_GCN_options_group(group_train=group_train)
    add_GP_options_group(group_train=group_train)

    # Niche trajectory
    add_NT_options_group(optparser=optparser)

    return optparser


def opt_gt_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_GT']  # type: ignore

    (options, args) = optparser.parse_args()

    # IO
    validate_io_options(optparser=optparser, options=options, io_options=io_options)
    # training
    validate_train_options(optparser=optparser, options=options)
    validate_GCN_options(optparser=optparser, options=options)
    validate_GP_options(optparser=optparser, options=options)
    # niche trajectory
    validate_NT_options(optparser=optparser, options=options)

    info('------------------ RUN params memo ------------------ ')
    # print parameters to stdout
    write_io_options_memo(options, io_options)
    write_train_options_memo(options)
    write_GCN_options_memo(options)
    write_GP_options_memo(options)
    write_NT_options_memo(options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


# ------------------------------------
# ONTraC_analysis functions
# ------------------------------------
def prepare_analysis_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_analysis']  # type: ignore

    usage = f'''USAGE: %prog <-o output> [--NN-dir PREPROCESSING_DIR] [--GNN-dir GNN_DIR] [--NT-dir NTSCORE_DIR] [--meta-input META_INPUT]
    [--scale-factor SCALE_FACTOR] [--reverse] [--suppress-cell-type-composition] [--suppress-niche-cluster-loadings] [--suppress-niche-trajectory]'''
    description = 'ONTraC_analysis: analysis of ONTraC results'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=io_options)
    add_visualization_group(optparser)
    add_suppress_group(optparser)

    return optparser


def opt_analysis_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: List[str] = IO_OPTIONS['ONTraC_analysis']  # type: ignore

    (options, args) = optparser.parse_args()

    validate_io_options(optparser=optparser,
                        options=options,
                        io_options=io_options,
                        required=False,
                        overwrite_validation=False)
    validate_visualization_options(options)
    validate_suppress_options(options)

    info('------------------ RUN params memo ------------------ ')
    # print parameters to stdout
    write_io_options_memo(options, io_options)
    write_visualization_options_memo(options)
    write_suppress_options_memo(options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


# ------------------------------------
# functions to be exported
# ------------------------------------
__all__ = [
    'prepare_ontrac_optparser', 'opt_ontrac_validate', 'prepare_nn_optparser', 'opt_nn_validate',
    'prepare_gnn_optparser', 'opt_gnn_validate', 'prepare_nt_optparser', 'opt_nt_validate', 'prepare_gt_optparser',
    'opt_gt_validate', 'prepare_analysis_optparser', 'opt_analysis_validate'
]
