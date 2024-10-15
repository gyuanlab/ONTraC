from optparse import OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *
from ._train import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def prepare_NT_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """
    usage = f'''USAGE: %prog <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> 
            [--trajectory-construct TRAJECTORY_CONSTRUCT]'''
    description = 'Niche trajectory: construct niche trajectory for niche cluster and project the NT score to each cell'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)
    add_NT_options_group(optparser=optparser)

    return optparser


def add_NT_options_group(optparser: OptionParser) -> None:
    """
    Add niche trajectory options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """

    group_NT = OptionGroup(optparser, "Options for niche trajectory")
    optparser.add_option_group(group_NT)
    group_NT.add_option(
        '--trajectory-construct',
        dest='trajectory_construct',
        default='BF',
        choices=['BF', 'TSP', 'DM'],
        help="Method to construct the niche trajectory. Choices: BF (brute force), TSP (Travelling salesman problem), DM (diffusion map). Default is 'BF' (brute-force).")


def write_NT_options_memo(options: Values) -> None:
    """
    Write niche trajectory options memo.
    :param options: Values, options.
    :return: None.
    """

    info('---------------- Niche trajectory options ----------------')
    info(f'Niche trajectory construction method: {options.trajectory_construct}')


def opt_NT_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser, options, IO_OPTIONS)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    write_NT_options_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
