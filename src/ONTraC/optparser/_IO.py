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
    if 'preprocessing_dir' in io_options:
        group_io.add_option('--preprocessing-dir',
                            dest='preprocessing_dir',
                            type='string',
                            help='Directory for preprocessing outputs.')
    if 'GNN_dir' in io_options:
        group_io.add_option('--GNN-dir', dest='GNN_dir', type='string', help='Directory for the GNN output.')
    if 'NTScore_dir' in io_options:
        group_io.add_option('--NTScore-dir', dest='NTScore_dir', type='string', help='Directory for the NTScore output.')
    if 'dataset' in io_options:
        group_io.add_option('-d', '--dataset', dest='dataset', type='string', help='Original input dataset.')

    optparser.add_option_group(group_io)


def validate_io_options(optparser: OptionParser,
                        options: Values,
                        io_options: Optional[List[str]],
                        overwrite_validation: bool = True) -> None:
    """Validate IO options from a OptParser object.
    :param optparser: OptionParser object.
    :param options: Options object.
    :param io_options: List of I/O options.
    :param overwrite_validation: Overwrite validation flag.
    :return: None.
    """
    if io_options is None:
        return
    if 'dataset' in io_options:
        if not options.dataset:
            error('Please provide a dataset.')
            optparser.print_help()
            sys.exit(1)
        if not os.path.isfile(options.dataset):
            error(f'The input file ({options.dataset}) you given does not exist.')
            optparser.print_help()
            sys.exit(1)
        if not options.dataset.endswith(('csv', 'csv.gz')):
            error(f'The input file ({options.dataset}) should be in csv format.')
            optparser.print_help()
            sys.exit(1)

    if 'preprocessing_dir' in io_options:
        if not options.preprocessing_dir:
            error('Please provide a directory for preprocessing outputs.')
            optparser.print_help()
            sys.exit(1)
        if os.path.isdir(options.preprocessing_dir):
            if overwrite_validation:
                warning(
                    f'The directory ({options.preprocessing_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.preprocessing_dir}')
            os.makedirs(options.preprocessing_dir, exist_ok=True)

    if 'GNN_dir' in io_options:
        if not options.GNN_dir:
            error('Please provide a directory for the GNN output.')
            optparser.print_help()
            sys.exit(1)
        if os.path.isdir(options.GNN_dir):
            if overwrite_validation:
                warning(f'The directory ({options.GNN_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.GNN_dir}')
            os.makedirs(options.GNN_dir, exist_ok=True)

    if 'NTScore_dir' in io_options:
        if not options.NTScore_dir:
            error('Please provide a directory for the NTScore output.')
            optparser.print_help()
            sys.exit(1)
        if os.path.isdir(options.NTScore_dir):
            if overwrite_validation:
                warning(f'The directory ({options.NTScore_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.NTScore_dir}')
            os.makedirs(options.NTScore_dir, exist_ok=True)


def write_io_options_memo(options: Values, io_options: Optional[List[str]]) -> None:
    """Write IO options to stdout.
    :param options: Options object.
    :param io_options: List of I/O options.
    :return: None.
    """
    if io_options is None:
        return
    info('            -------- I/O options -------             ')
    if 'preprocessing_dir' in io_options:
        info(f'preprocessing output directory:  {options.preprocessing_dir}')
    if 'GNN_dir' in io_options:
        info(f'GNN output directory:  {options.GNN_dir}')
    if 'NTScore_dir' in io_options:
        info(f'NTScore output directory:  {options.NTScore_dir}')
    if 'dataset' in io_options:
        info(f'dataset: {options.dataset}')
