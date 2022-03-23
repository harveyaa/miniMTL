#!/bin/bash
log_dir='/home/harveyaa/Documents/masters/MTL/miniMTL/examples/arch_search/logs'
script='../model_00.py'

################
# SINGLE TASKS #
################
#singles='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'
singles='SZ ASD BIP DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2'

for single in $singles
do
    #python $script --tasks $single --num_epochs 20 --log_dir $log_dir
    echo $single
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
        python $script --tasks $big $small --num_epochs 50 --log_dir $log_dir
    done
done

#############
# SUMMARIZE #
#############
python summarize_results.py --log_dir $log_dir