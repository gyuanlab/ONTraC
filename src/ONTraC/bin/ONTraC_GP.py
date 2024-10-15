#!/usr/bin/env python

import os
import sys
from typing import Callable, Optional

from ..model import GraphPooling
from ..optparser import opt_GP_validate, prepare_GP_optparser
from ..run.processes import NTScore, gnn, load_parameters
from ..train import GPBatchTrain
from ..train.inspect_funcs import loss_record
from ..utils import read_original_data, valid_original_data, write_version_info

# ------------------------------------
# Classes
# ------------------------------------


# ------------------------------------
# Functions
# ------------------------------------
def get_inspect_funcs() -> Optional[list[Callable]]:
    """
    Inspect function list
    :param output_dir: output dir
    :param epoch_filter: epoch filter
    :return: list of inspect functions
    """
    return [loss_record]


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """
    Main function
    :return: None
    """

    # write version information
    write_version_info()

    # load parameters
    options = load_parameters(opt_validate_func=opt_GP_validate, prepare_optparser_func=prepare_GP_optparser)
    
    # load original data
    options.dataset = os.path.join(f'{options.preprocessing_dir}', 'original_data.csv')
    if not os.path.exists(options.dataset):
        raise FileNotFoundError(f"Dataset file not found: {options.dataset}. You may need to run createDataSet first or copy original dataset file into {options.preprocessing_dir} directory with the name 'original_data.csv'.")
    
    ori_data_df = read_original_data(options=options)
    ori_data_df = valid_original_data(ori_data_df=ori_data_df)

    # ----- GNN -----
    gnn(options=options,
        ori_data_df=ori_data_df,
        nn_model=GraphPooling,
        BatchTrain=GPBatchTrain,
        inspect_funcs=get_inspect_funcs())

    # ----- NT score -----
    NTScore(options=options)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
