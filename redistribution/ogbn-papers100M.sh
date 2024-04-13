ip_list=ip_list.txt
feat_group=1
graph_group=$1
fanout=50


# Don't change below
graph_name="ogbn-papers100M"
edge_file="~/efs/raw_datasets/ogbn-papers100M/edge_list.parquet"
# node_file="~/efs/raw_datasets/ogbn-papers100M/features.parquet"
node_file="~/efs/raw_datasets/ogbn-papers100M/8_parts/features.parquet"
# num_node_files=16
# num_edge_files=16
num_node_files=8
num_edge_files=8

# for graph_group in {1..16}; do
#     for feat_group in {1..16}; do
#         if (( graph_group * feat_group == 16 )); then
#             for model in gcn gat; do
#                 bash launch.sh \
#                 $ip_list \
#                 $feat_group \
#                 $graph_group \
#                 $graph_name \
#                 $edge_file \
#                 $node_file \
#                 $num_node_files \
#                 $num_edge_files \
#                 $model \
#                 ring \
#                 $fanout
#                 if [ $? -ne 0 ]; then
#                     echo "Error in launching"
#                     exit 1
#                 fi
#             done
#         fi
#     done
# done

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
