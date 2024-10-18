from optparse import OptionGroup, OptionParser, Values

from ..log import *
from ..version import __version__
from ._create_dataset import (add_niche_net_constr_options_group,
                              validate_niche_net_constr_options,
                              write_niche_net_constr_memo)
from ._IO import *
from ._NT import add_NT_options_group, write_NT_options_memo
from ._train import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['dataset', 'preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def prepare_ontrac_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in this function first.
    :return: OptionParser object.
    """
    usage = f'''USAGE: %prog <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> <-d DATASET>
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR]
    [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT]
    [--purity-loss-weight PURITY_LOSS_WEIGHT] [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]
    [--degree-out-adj-norm] [--expectation-out-adj-norm] [--trajectory-construct TRAJECTORY_CONSTRUCT]'''
    description = 'All steps of ONTraC including dataset creation, Graph Pooling, and NT score calculation.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    # Niche net construction option group
    add_niche_net_constr_options_group(optparser)

    # Graph Pooling
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    # Niche trajectory
    add_NT_options_group(optparser)

    return optparser


def opt_ontrac_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """
    (options, args) = optparser.parse_args()

    # IO
    validate_io_options(optparser=optparser, options=options, io_options=IO_OPTIONS)
    # niche network construction
    validate_niche_net_constr_options(optparser, options)
    # training
    validate_train_options(optparser, options)
    validate_NP_options(optparser, options)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    write_niche_net_constr_memo(options)
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    write_NT_options_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
