#!/usr/bin/env python

import sys

from ..optparser import opt_gnn_validate, prepare_gnn_optparser
from ..run.processes import gnn, load_parameters
from ..utils import write_version_info


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """
    The main function
    """

    # write version information
    write_version_info()

    # load parameters
    options = load_parameters(opt_validate_func=opt_gnn_validate, prepare_optparser_func=prepare_gnn_optparser)

    # ----- GNN -----
    gnn(options=options)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
