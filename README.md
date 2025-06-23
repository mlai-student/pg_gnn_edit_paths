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

### Using pip with requirements.txt

Alternatively, you can set up the environment manually using pip and the provided requirements.txt file:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

This will install all the required dependencies including NumPy, Matplotlib, NetworkX, PyTorch CPU version, PyTorch Geometric (PyG), and the Open Graph Benchmark (OGB) package.

