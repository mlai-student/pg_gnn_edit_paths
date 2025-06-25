# GNN Edit Paths

This repository serves as a starting point for a project that explores graph edit paths using Graph Neural Networks (GNNs).

## Overview

This project focuses on generating and analyzing edit paths between graphs. An edit path represents a sequence of operations (node/edge additions, deletions, or relabelings) that transform one graph into another. The key components of this project include:

1. **Graph Dataset Loading**: Load graph datasets from various sources using PyTorch Geometric
2. **Edit Path Generation**: Generate edit paths between pairs of graphs in a dataset
3. **Intermediate Graph Creation**: Create intermediate graphs along an edit path
4. **Visualization**: Visualize graphs and edit paths

These components provide a foundation for further research on graph edit distances and graph similarity using Graph Neural Networks.


## Installation

### Using the installation script

To set up the environment, you can use the provided installation script:

```bash
# Make the script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

The script will:
1. Detect your operating system
2. Create a Python virtual environment
3. Install all required dependencies including:
   - NumPy, Matplotlib, NetworkX
   - PyTorch CPU version
   - PyTorch Geometric (PyG)
   - Open Graph Benchmark (OGB) package

After installation, you can activate the environment using:
- On Windows: `venv\Scripts\activate`
- On macOS/Linux: `source venv/bin/activate`

## Usage

### generate_path_graphs.py

The `generate_path_graphs.py` script (formerly main.py) demonstrates the core functionality of this repository:

1. Loads a graph dataset using the GraphDataset class
2. Creates NetworkX graph representations of the dataset
3. Generates pairwise edit paths between graphs and saves them to a file

To run the script with default parameters:

```bash
python generate_path_graphs.py
```

You can customize the execution with the following command-line arguments:

```bash
python generate_path_graphs.py --db_name MUTAG --optimization_iterations 100 --timeout 60 --max_workers 4 --output_dir data
```

Available arguments:
- `--db_name`: Name of the database/dataset to use (default: MUTAG)
- `--optimization_iterations`: Number of optimization iterations (default: 100)
- `--timeout`: Timeout in seconds for each graph pair processing (default: 60)
- `--max_workers`: Maximum number of worker processes (default: None for auto-detection)
- `--output_dir`: Directory to store output files (default: data)

### load_path_graphs.py

The `load_path_graphs.py` script allows you to load previously generated edit paths between specific graphs:

1. Loads a graph dataset using the GraphDataset class
2. Creates NetworkX graph representations of the dataset
3. Loads edit paths between specified graphs
4. Creates intermediate graphs for the edit path

To run the script with default parameters:

```bash
python load_path_graphs.py
```

You can customize the execution with the following command-line arguments:

```bash
python load_path_graphs.py --db_name MUTAG --data_dir data --start_graph_id 0 --end_graph_id 1 --seed 42
```

Available arguments:
- `--db_name`: Name of the database/dataset to use (default: MUTAG)
- `--data_dir`: Directory where the edit paths are stored (default: data)
- `--start_graph_id`: ID of the start graph (default: 0)
- `--end_graph_id`: ID of the end graph (default: 1)
- `--seed`: Seed for reproducibility in plotting (default: 42)

## Project Structure

The repository is organized as follows:

### Main Scripts
- `generate_path_graphs.py`: Generates edit paths between graphs in a dataset
- `load_path_graphs.py`: Loads previously generated edit paths and creates intermediate graphs
- `check_packages.py`: Utility script to check if all required packages are installed

### Directories
- `data/`: Contains datasets and generated edit paths
- `example_paths_MUTAG/`: Example edit paths for the MUTAG dataset
- `utils/`: Utility modules for the project
  - `EditPath.py`: Implementation of the EditPath class for representing and manipulating edit paths
  - `generate_edit_paths.py`: Functions for generating edit paths between graphs
  - `io.py`: I/O functions for loading and saving edit paths
  - `plotting.py`: Functions for plotting graphs and edit paths
  - `GraphLoader/`: Module for loading graph datasets from various sources

### Configuration Files
- `requirements.txt`: List of Python package dependencies
- `install.sh`: Installation script for setting up the environment
