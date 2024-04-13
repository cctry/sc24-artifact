#!/bin/bash

readarray -t IPS < $1
# N=${#IPS[@]}
feat_group=$2
graph_group=$3
N=$(( $2 * $3 ))

# choose the first machine as master
MASTER=${IPS[0]}
PORT=17899  # choose an available port

source_id_col="src"
dest_id_col="dst"
node_id_col="id"
feature_col="feat"

graph_name=$4
edge_file=$5
node_file=$6
num_node_files=$7
num_edge_files=$8
model=$9
scheduler=${10}
fanout=${11}

for ((i=0;i<N;i++)); do
    IP=${IPS[i]}
    # echo "Launching process $i on $IP"
    ssh -t -t -o StrictHostKeyChecking=no $IP \
    -i ~/.ssh/csy_w1.pem \
    "cd $PWD; \
    source activate dgl;  \
    torchrun \
    --nproc_per_node=1 \
    --nnodes=$N \
    --node_rank=$i \
    --master_addr=$MASTER \
    --master_port=$PORT \
    main.py --source_id_col $source_id_col --dest_id_col $dest_id_col \
    --edge_file $edge_file --node_id_col $node_id_col \
    --feature_col $feature_col \
    --node_file $node_file --graph_name $graph_name\
    --feat_group $feat_group --graph_group $graph_group\
    --num_node_files $num_node_files --num_edge_files $num_edge_files\
    --model $model --scheduler $scheduler --fanout $fanout\
    " &
done
wait

# Initialize the output filename
output_file="${graph_name}_${model}_${scheduler}_${fanout}_${feat_group}_${graph_group}.log"
output_file_path="results/$output_file"
# Clear the content of the output file if it exists
> $output_file_path

echo $output_file

# Loop through each log file and append its contents to the output file
for i in $(ls *_${output_file}); do
  cat "$i" >> $output_file_path
  rm "$i"
done

