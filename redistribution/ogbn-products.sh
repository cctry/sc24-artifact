ip_list=ip_list.txt
feat_group=1
graph_group=$1
fanout=50


# Don't change below
graph_name=ogbn-products
edge_file="~/efs/raw_datasets/ogbn-products/edge_list.parquet"
node_file="~/efs/raw_datasets/ogbn-products/features.parquet"
num_node_files=8
num_edge_files=8

bash launch.sh \
$ip_list \
$feat_group \
$graph_group \
$graph_name \
$edge_file \
$node_file \
$num_node_files \
$num_edge_files \
gat \
ring \
$fanout