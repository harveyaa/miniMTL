#!/bin/bash
log_dir='/home/harveyaa/Documents/masters/MTL/miniMTL/examples/arch_search/model_00/logs'
script='/home/harveyaa/Documents/masters/MTL/miniMTL/examples/arch_search/arch_search.py'
summarize_pairs='/home/harveyaa/Documents/masters/MTL/miniMTL/examples/arch_search/summarize_pairs.py'
encoder=1
head=2

################
# SINGLE TASKS #
################
#singles='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'
singles='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2'

for single in $singles
do
    python $script --tasks $single --num_epochs 20 --log_dir $log_dir --encoder $encoder --head $head
done

################
# PAIRED TASKS #
################

#bigs = 'SZ ASD BIP'
#smalls = 'DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

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