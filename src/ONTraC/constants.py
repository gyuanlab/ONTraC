from typing import Dict, List

IO_OPTIONS: Dict[str, List[str]] = {
    'ONTraC': ['input', 'NN_dir', 'GNN_dir', 'NT_dir'],
    'ONTraC_NN': ['input', 'NN_dir'],  # niche network
    'ONTraC_GNN': ['NN_dir', 'GNN_dir'],  # graph neural network
    'ONTraC_NT': ['NN_dir', 'GNN_dir', 'NT_dir'],  # niche trajectory
    'ONTraC_GT': ['NN_dir', 'GNN_dir', 'NT_dir'],  # GNN & niche trajectory
    'ONTraC_analysis': ['input', 'NN_dir', 'GNN_dir', 'NT_dir', 'output', 'log'],  # analysis
}
