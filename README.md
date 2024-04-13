# SC24-artifact
This repo contains the artifact system of a paper submmited to SC24.

## Usage
### Installation
Under the repo directory.
> pip install -e .
### Graph preprocessing
Deal requires to preprocess the datasets into Parquet files. 
In particular, a graph can be represented in a set of edge files and node feature files. 
The node file and edge file the following columns.
#### Node file
1. **ID**: The column to store the node ID in int64.
2. **feat0**: The column to store the node feature as a list of float.
Deal supports to further split the feature vector into small segements to enabling feature partition. The partitioned features needs to be in the column named as **feat1**, **feat2**, and etc.
#### Edge file
1. **src**: The source node ID.
2. **dst**: The destination node ID.

Note that Deal supports to chunk the Parquet files for parallelized file reading and reduced memory usage. The chunked edge files should be named like `edges_i.parquet`, and the node files are named like `features_i_split.parquet`.

The example script is provided in `Dataset/ogb_dataset.py`. This script will preprocess graphs from Open Graph Benchmark.
### Run infernece
Deal is built on PyTorch DDP so it can be lanuched through torchrun or distributed launcher.
The example script is availiable `real_graph/ogbn-papers100M.sh`. Note that the example lanuch Deal using torchrun. It requires the master machine has the access to all working machines without password needed.
