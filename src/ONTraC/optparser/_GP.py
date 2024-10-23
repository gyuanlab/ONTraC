from optparse import OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *
from ._NT import add_NT_options_group, write_NT_options_memo
from ._train import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def prepare_GP_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """
    usage = f'''USAGE: %prog <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>
    [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--degree-out-adj-norm] [--expectation-out-adj-norm]
    [--trajectory-construct TRAJECTORY_CONSTRUCT]'''
    description = 'ONTraC_GP: GNN and Niche Trajectory'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    # train and model options
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    # Niche trajectory
    add_NT_options_group(optparser)

    return optparser


def opt_GP_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser, options, IO_OPTIONS)
    validate_train_options(optparser, options)
    validate_GNN_options(optparser, options)
    validate_NP_options(optparser, options)

    info('------------------ RUN params memo ------------------ ')
    # print parameters to stdout
    write_io_options_memo(options, IO_OPTIONS)
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    write_NT_options_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
