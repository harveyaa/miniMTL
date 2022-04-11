#!/bin/bash
#########
# USAGE #
#########
# e.g. to run for encoder 0 head 1 with 3 tasks:
# ./local_groups.sh 0 1 3 /path/to/dir/

encoder=$1
head=$2
n_tasks=$3
log_dir="$4model_$encoder$head"
echo "Logging to..."
echo $log_dir
mkdir $log_dir

base_dir="/home/aharvey/MTL/miniMTL/examples/arch_search"
script="/home/aharvey/MTL/miniMTL/examples/hps.py"
summarize_groups="$base_dir/groups/summarize_groups.py"
data_dir='/home/aharvey/MTL/data'

tasks='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

echo "encoder: $encoder" >> "$log_dir/config.txt"
echo "head: $head" >> "$log_dir/config.txt"
echo "n_tasks: $n_tasks" >> "$log_dir/config.txt"
echo "tasks: $tasks" >> "$log_dir/config.txt"

################
# SINGLE TASKS #
################
for task in $tasks
do
    python $script --tasks $task --num_epochs 50 --log_dir $log_dir --encoder $encoder --head $head --data_dir $data_dir
done

#################
# GROUPED TASKS #
#################
python - << EOF
import itertools
cases = ['SZ', 'ASD', 'BIP', 'DEL22q11_2']# 'DUP22q11_2', 'DEL16p11_2', 'DUP16p11_2']
combos = [c for c in itertools.combinations(cases,$n_tasks)]

with open('temp_combos.txt', 'w') as file:
    for c in combos:
        string = ' '.join(c)
        file.write(f"{string}\n")
EOF

filename="temp_combos.txt"

while IFS= read -r line; do
    python $script --tasks $line --num_epochs 100 --log_dir $log_dir --encoder $encoder --head $head --data_dir $data_dir
done < $filename

rm temp_combos.txt

#############
# SUMMARIZE #
#############
python $summarize_groups --log_dir $log_dir --n_tasks $n_tasks