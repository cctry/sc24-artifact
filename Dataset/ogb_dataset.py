import dgl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from ogb.nodeproppred import DglNodePropPredDataset

def download_and_load_graph(dataset_name):
    dataset = DglNodePropPredDataset(name=dataset_name, root='dataset/')
    graph, _ = dataset[0]
    return graph

def export_edges_to_parquet(graph, num_edge_files, parquet_dir):
    src, dst = graph.edges()
    src = src.numpy()
    dst = dst.numpy()
    num_edges = src.shape[0]
    edges_per_file = num_edges // num_edge_files

    for i in range(num_edge_files):
        start_idx = i * edges_per_file
        end_idx = (start_idx + edges_per_file) if i < num_edge_files - 1 else num_edges
        table = pa.Table.from_arrays([src[start_idx:end_idx], dst[start_idx:end_idx]], names=['src', 'dst'])
        pq.write_table(table, f"{parquet_dir}/edge_list_{i}.parquet")

def export_node_features_to_parquet(graph, num_feature_files, parquet_dir):
    features = graph.ndata['feat'].numpy()
    node_ids = graph.ndata[dgl.NID].numpy()
    num_nodes = features.shape[0]
    num_features = features.shape[1]
    rows_per_file = num_nodes // num_feature_files

    for i in range(num_feature_files):
        start_idx = i * rows_per_file
        end_idx = (start_idx + rows_per_file) if i < num_feature_files - 1 else num_nodes
        feature_columns = [pa.array(features[start_idx:end_idx, j]) for j in range(num_features)]
        node_id_column = pa.array(node_ids[start_idx:end_idx])
        columns = [node_id_column] + feature_columns
        column_names = ['ID'] + [f'feat{j}' for j in range(num_features)]
        table = pa.Table.from_arrays(columns, names=column_names)
        pq.write_table(table, f"{parquet_dir}/features_{i}_split.parquet")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Export Graph Edges and Features to Parquet")
    parser.add_argument("--dataset", type=str, required=True, help="OGB dataset name")
    parser.add_argument("--edge_files", type=int, default=1, help="Number of edge parquet files")
    parser.add_argument("--feature_files", type=int, default=1, help="Number of feature parquet files")
    parser.add_argument("--output_dir", type=str, default="output_parquet", help="Output directory for Parquet files")
    return parser.parse_args()

def main():
    args = parse_arguments()
    graph = download_and_load_graph(args.dataset)
    export_edges_to_parquet(graph, args.edge_files, args.output_dir)
    export_node_features_to_parquet(graph, args.feature_files, args.output_dir)

if __name__ == '__main__':
    main()
