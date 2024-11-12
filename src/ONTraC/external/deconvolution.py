import subprocess
from importlib import resources
from pathlib import Path
from typing import Union
from ..log import info

import numpy as np
import pandas as pd
import gzip


def apply_STdeconvolve(NN_dir: Union[str, Path], exp_matrix: np.ndarray, ct_num: int) -> np.ndarray:
    """
    Apply STdeconvolve to spot-level data.
    :param NN_dir: Niche network directory.
    :param exp_matrix: Expression matrix.  #spot x #gene
    :param ct_num: Number of cell types.
    :return: Deconvoluted expression matrix.  #spot x #cell_type
    """

    # save expression matrix in csv format
    exp_matrix_file = f'{NN_dir}/filtered_spot_exp.csv.gz'
    exp_matrix.to_csv(exp_matrix_file, sep=',', index=True, header=True, compression='gzip')

    spot_x_cell_type_file = '/spot_x_celltype_deconvolution.csv.gz'

    with resources.path("ONTraC.external", "STdeconvolve.R") as stdeconvolve_script_path:
        try:
            stdeconvolve_script_path_str = str(stdeconvolve_script_path)
            subprocess.run(["Rscript", stdeconvolve_script_path_str, exp_matrix_file, str(ct_num), NN_dir, spot_x_cell_type_file])
        except subprocess.CalledProcessError as e:
            print("Error in running R script:", e)
            print("R script stderr:", e.stderr)
    info('Finished cell type deconvolution.')

    # load deconvoluted cell type composition data
    deconvoluted_ct_matrix = np.loadtxt(f'{NN_dir}/spot_x_celltype_deconvolution.csv.gz', delimiter=',')
    return deconvoluted_ct_matrix
