#!/bin/bash
#########
# USAGE #
#########
# e.g. to run for encoder 0 head 1:
# ./local_groups.sh 0 1

encoder=$1
head=$2

base_dir="/home/harveyaa/Documents/masters/MTL/miniMTL/examples/arch_search/pairs"
script="/home/harveyaa/Documents/masters/MTL/miniMTL/examples/hps.py"
summarize_pairs="$base_dir/summarize_pairs.py"

log_dir="$base_dir/model_$encoder$head"
mkdir $log_dir

#tasks='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'
tasks='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2'

echo "encoder: $encoder" >> "$log_dir/config.txt"
echo "head: $head" >> "$log_dir/config.txt"
echo "n_tasks: $n_tasks" >> "$log_dir/config.txt"
echo "tasks: $tasks" >> "$log_dir/config.txt"

################
# SINGLE TASKS #
################
for task in $tasks
do
    python $script --tasks $task --num_epochs 20 --log_dir $log_dir --encoder $encoder --head $head
done

################
# PAIRED TASKS #
################

bigs='SZ ASD BIP'
smalls='DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2'

for big in $bigs
do
    for small in $smalls
    do
        python $script --tasks $big $small --num_epochs 50 --log_dir $log_dir --encoder $encoder --head $head
    done
done

#############
# SUMMARIZE #
#############
python $summarize_pairs --log_dir $log_dir