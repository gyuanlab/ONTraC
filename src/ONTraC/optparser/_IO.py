import os
import sys
from optparse import OptionGroup, OptionParser, Values
from typing import List, Optional

from ..log import *


def add_IO_options_group(optparser: OptionParser, io_options: Optional[List[str]]) -> None:
    """
    Add I/O options group to optparser.
    :param optparser: OptionParser object.
    :param io_options: List of I/O options.
    :return: OptionGroup object.
    """
    if io_options is None:
        return
    # I/O options group
    group_io = OptionGroup(optparser, "IO")

    # directories
    if 'NN_dir' in io_options:
        group_io.add_option('--NN-dir', dest='NN_dir', type='string', help='Directory for niche network outputs.')
    if 'GNN_dir' in io_options:
        group_io.add_option('--GNN-dir', dest='GNN_dir', type='string', help='Directory for the GNN output.')
    if 'NT_dir' in io_options:
        group_io.add_option('--NT-dir', dest='NT_dir', type='string', help='Directory for the niche trajectory output.')
    if 'output' in io_options:
        group_io.add_option('-o', '--output', dest='output', type='string', help='Directory for analysis output.')

    # input files
    if 'input' in io_options:
        group_io.add_option(
            '--meta-input',
            dest='meta_input',
            type='string',
            help=
            'Meta data file in csv format. Each row is a cell and each column is a meta data. The first column should be the cell name. Coordinates (x, y) and sample should be included. Cell type is required for cell-level data.'
        )
        group_io.add_option('--low-res-exp-input',
                            dest='low_res_exp_input',
                            type='string',
                            help='Gene X spot matrix in csv format for low-resolution dataset.')
    if 'log' in io_options:
        group_io.add_option('-l', '--log', dest='log', type='string', help='Log file.')

    # deprecated options
    if 'NN_dir' in io_options:
        group_io.add_option('--preprocessing-dir',
                            dest='preprocessing_dir',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --NN-dir instead.')
    if 'NT_dir' in io_options:
        group_io.add_option('--NTScore-dir',
                            dest='NTScore_dir',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --NT-dir instead.')
    if 'input' in io_options:
        group_io.add_option('-d',
                            '--dataset',
                            dest='dataset',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --meta-input instead.')

    optparser.add_option_group(group_io)


def validate_io_options(optparser: OptionParser,
                        options: Values,
                        io_options: Optional[List[str]],
                        required: bool = True,
                        overwrite_validation: bool = True) -> None:
    """Validate IO options from a OptParser object.
    :param optparser: OptionParser object.
    :param options: Options object.
    :param io_options: List of I/O options.
    :param required: Required flag.
    :param overwrite_validation: Overwrite validation flag.
    :return: None.
    """
    if io_options is None:
        return
    if 'input' in io_options:

        # meta data
        if hasattr(options, 'dataset') and options.dataset is not None and (not hasattr(
                options, 'meta_input') or options.meta_input is None):  # with dataset and without meta_input
            warning('The --dataset option will be deprecated from v3.0. Please use --meta-input instead.')
            options.meta_input = options.dataset
        if required and (not hasattr(options, 'meta_input') or options.meta_input is None):
            error('Please provide a meta data file in csv format.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'meta_input') and options.meta_input is not None and not os.path.isfile(options.meta_input):
            error(f'The meta data file ({options.meta_input}) you given does not exist.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'meta_input') and options.meta_input is not None and not options.meta_input.endswith(
            ('csv', 'csv.gz')):
            error(f'The meta data file ({options.meta_input}) should be in csv format.')
            optparser.print_help()
            sys.exit(1)

        # low resolution expression data
        if hasattr(options, 'low_res_exp_input') and options.low_res_exp_input is not None:
            if not os.path.isfile(options.low_res_exp_input):
                error(f'The expression data file ({options.low_res_exp_input}) you given does not exist.')
                optparser.print_help()
                sys.exit(1)
            if not options.low_res_exp_input.endswith(('csv', 'csv.gz')):
                error(f'The expression data file ({options.low_res_exp_input}) should be in csv format.')
                optparser.print_help()
                sys.exit(1)

    if 'NN_dir' in io_options:
        if hasattr(options,
                   'preprocessing_dir') and options.preprocessing_dir is not None and (not hasattr(options, 'NN_dir')
                                                                                       or options.NN_dir is None):
            warning('The --preprocessing-dir option will be deprecated from v3.0. Please use --NN-dir instead.')
            options.NN_dir = options.preprocessing_dir
        if required and (not hasattr(options, 'NN_dir') or options.NN_dir is None):
            error('Please provide a directory for niche network outputs.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'NN_dir') and options.NN_dir is not None and os.path.isdir(options.NN_dir):
            if overwrite_validation:
                warning(f'The directory ({options.NN_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.NN_dir}')
            os.makedirs(options.NN_dir, exist_ok=True)

    if 'GNN_dir' in io_options:
        if required and (not hasattr(options, 'GNN_dir') or options.GNN_dir is None):
            error('Please provide a directory for the GNN output.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'GNN_dir') and options.GNN_dir is not None and os.path.isdir(options.GNN_dir):
            if overwrite_validation:
                warning(f'The directory ({options.GNN_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.GNN_dir}')
            os.makedirs(options.GNN_dir, exist_ok=True)

    if 'NT_dir' in io_options:
        if hasattr(options, 'NTScore_dir') and options.NTScore_dir is not None and (not hasattr(options, 'NT_dir')
                                                                                    or options.NT_dir is None):
            warning('The --NTScore-dir option will be deprecated from v3.0. Please use --NT-dir instead.')
            options.NT_dir = options.NTScore_dir
        if required and (not hasattr(options, 'NT_dir') or options.NT_dir is None):
            error('Please provide a directory for the NTScore output.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'NT_dir') and options.NT_dir is not None and os.path.isdir(options.NT_dir):
            if overwrite_validation:
                warning(f'The directory ({options.NT_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.NT_dir}')
            os.makedirs(options.NT_dir, exist_ok=True)

    if 'output' in io_options:
        if not options.output:
            error('Please provide a directory for analysis output.')
            optparser.print_help()
            sys.exit(1)
        if os.path.isdir(options.output):
            if overwrite_validation:
                warning(f'The directory ({options.output}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.output}')
            os.makedirs(options.output, exist_ok=True)

    if 'log' in io_options:
        if hasattr(options, 'log') and options.log is not None and not os.path.isfile(options.log):
            error(f'Log file: {options.log} you given does not exist.')
            sys.exit(1)


def write_io_options_memo(options: Values, io_options: Optional[List[str]]) -> None:
    """Write IO options to stdout.
    :param options: Options object.
    :param io_options: List of I/O options.
    :return: None.
    """
    if io_options is None:
        return
    info('            -------- I/O options -------             ')
    if 'NN_dir' in io_options and hasattr(options, 'NN_dir') and options.NN_dir is not None:
        info(f'Niche network output directory:  {options.NN_dir}')
    if 'GNN_dir' in io_options and hasattr(options, 'GNN_dir') and options.GNN_dir is not None:
        info(f'GNN output directory:  {options.GNN_dir}')
    if 'NT_dir' in io_options and hasattr(options, 'NT_dir') and options.NT_dir is not None:
        info(f'Niche trajectory output directory:  {options.NT_dir}')
    if 'output' in io_options:
        info(f'Output directory:  {options.output}')
    if 'input' in io_options and hasattr(options, 'meta_input') and options.meta_input is not None:
        info(f'Meta data file:  {options.meta_input}')
        if hasattr(options, 'low_res_exp_input') and options.low_res_exp_input is not None:
            info(f'Low resolution expression data file:  {options.low_res_exp_input}')
    if 'log' in io_options and options.log is not None:
        if hasattr(options, 'log'):
            info(f'Log file:  {options.log}')
