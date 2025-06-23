# GNN Edit Paths

This repository serves as starting point for a project that explores graph edit paths using Graph Neural Networks (GNNs).


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

### main.py

The `main.py` script demonstrates the core functionality of this repository:

1. Loads the MUTAG dataset using the GraphDataset class
2. Creates NetworkX graph representations of the dataset
3. Generates pairwise edit paths between graphs and saves them to a file
4. Loads the edit paths from the file
5. Creates intermediate graphs for an edit path using a valid random permutation of operations

To run the script:

```bash
python main.py
```
