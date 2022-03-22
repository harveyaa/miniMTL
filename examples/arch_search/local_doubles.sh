#!/bin/bash

#bigs = 'SZ ASD BIP'
#smalls = 'DEL22q11_2 DUP22q11_2 DEL16p11_2 DUP16p11_2 DEL1q21_1 DUP1q21_1'

bigs='BIP'
smalls='DEL22q11_2 DUP22q11_2'

for big in $bigs
do
    for small in $smalls
    do
        python ../n_task.py --tasks $big $small --num_epochs 100
    done
done