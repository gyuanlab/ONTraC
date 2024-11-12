import os
import sys
from optparse import OptionGroup, OptionParser, Values
from typing import List, Optional

from ..log import *


def add_visualization_group(optparser: OptionParser) -> None:
    group = OptionGroup(optparser, 'Visualization options')
    group.add_option('-r',
                         '--reverse',
                         dest='reverse',
                         action='store_true',
                         default=False,
                         help='Reverse the NT score during visualization.')
    group.add_option('-s',
                     '--sample',
                     dest='sample',
                     action='store_true',
                     default=False,
                     help='Plot each sample separately.')
    group.add_option('--scale-factor',
                     dest='scale_factor',
                     type='float',
                     default=1.0,
                     help='Scale factor control the size of spatial-based plots.')
    optparser.add_option_group(group)


def add_suppress_group(optparser: OptionParser) -> None:
    group = OptionGroup(optparser, 'Suppress options')
    group.add_option('--suppress-cell-type-composition',
                     dest='suppress_cell_type_composition',
                     action='store_true',
                     default=False,
                     help='Suppress the cell type composition visualization.')
    group.add_option('--suppress-niche-cluster-loadings',
                     dest='suppress_niche_cluster_loadings',
                     action='store_true',
                     default=False,
                     help='Suppress the niche cluster loadings visualization.')
    group.add_option('--suppress-niche-trajectory',
                     dest='suppress_niche_trajectory',
                     action='store_true',
                     default=False,
                     help='Suppress the niche trajectory related visualization.')
    optparser.add_option_group(group)


def validate_visualization_options(options: Values) -> None:
    """
    Validate visualization options.
    Placeholder and do nothing.

    :param options: Values object.
    :return: None.
    """

    pass


def validate_suppress_options(options: Values) -> None:
    """
    Validate suppress options.
    Placeholder and do nothing.

    :param options: Values object.
    :return: None.
    """

    pass


def write_visualization_options_memo(options: Values) -> None:
    """
    Write visualization options memo.
    :param options: Values object.
    :return: None.
    """

    info(message='---------------- Visualization options ----------------')
    info(message=f'Reverse the NT score during visualization: {options.reverse}')
    info(message=f'Plot each sample separately: {options.sample}')
    info(message=f'Scale factor control the size of spatial-based plots: {options.scale_factor}')


def write_suppress_options_memo(options: Values) -> None:
    """
    Write suppress options memo.
    :param options: Values object.
    :return: None.
    """

    info(message='---------------- Suppress options ----------------')
    info(message=f'Suppress the cell type composition visualization: {options.suppress_cell_type_composition}')
    info(message=f'Suppress the niche cluster loadings visualization: {options.suppress_niche_cluster_loadings}')
    info(message=f'Suppress the niche trajectory related visualization: {options.suppress_niche_trajectory}')
