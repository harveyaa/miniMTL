import os
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from miniMTL.datasets import caseControlDataset
from miniMTL.models import *
from miniMTL.training import Trainer
from miniMTL.hps import HPSModel
from miniMTL.util import split_data

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tasks",help="tasks",dest='tasks',nargs='*')
    parser.add_argument("--encoder",help="Which encoder to use.",dest='encoder',default=0,type=int)
    parser.add_argument("--head",help="Which head to use.",dest='head',default=0,type=int)
    parser.add_argument("--id_dir",help="path to data ods",dest='id_dir',
                        default='/home/harveyaa/Documents/masters/MTL/conf_balancing/hybrid/')
    parser.add_argument("--data_dir",help="path to data dir",dest='data_dir',
                        default='/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/')
    parser.add_argument("--data_format",help="data format code",dest='data_format',default=1,type=int)
    parser.add_argument("--log_dir",help="path to log_dir",dest='log_dir',default=None)
    parser.add_argument("--batch_size",help="batch size for training/test loaders",default=16,type=int)
    parser.add_argument("--lr",help="learning rate for training",default=1e-3,type=float)
    parser.add_argument("--num_epochs",help="num_epochs for training",default=100,type=int)
    parser.add_argument("--fold",help="fold of CV",default=0,type=int)
    parser.add_argument("--rand_test",help="use random test sets.",action='store_true')
    args = parser.parse_args()
    
    print('#############\n# HPS model #\n#############')
    print('Task(s): ',args.tasks)
    print('Encoder: ',args.encoder)
    print('Head(s): ',args.head)
    print('Batch size: ',args.batch_size)
    print('LR: ', args.lr)
    print('Epochs: ', args.num_epochs)
    print('#############\n')

    # Define paths to data
    p_pheno = os.path.join(args.data_dir,'pheno_01-12-21.csv')
    p_ids = args.id_dir
    p_conn = os.path.join(args.data_dir,'connectomes')

    # Create datasets
    print('Creating datasets...')
    cases = args.tasks
    data = []
    for case in cases:
        print(case)
        data.append(caseControlDataset(case,p_pheno,ids_path=p_ids,conn_path=p_conn,type='conn',strategy='balanced',format=args.data_format))
    print('Done!\n')
    
    if args.rand_test:
        # Split data & create loaders & loss fns
        loss_fns = {}
        trainloaders = {}
        testloaders = {}
        decoders = {}
        for d, case in zip(data,cases):
            train_d, test_d = split_data(d)

            trainloaders[case] = DataLoader(train_d, batch_size=args.batch_size, shuffle=True)
            testloaders[case] = DataLoader(test_d, batch_size=args.batch_size, shuffle=True)
            loss_fns[case] = nn.CrossEntropyLoss()
            decoders[case] = eval(f'head{args.head}().double()')
    else:
        # Split data & create loaders & loss fns
        loss_fns = {}
        trainloaders = {}
        testloaders = {}
        decoders = {}
        for d, case in zip(data,cases):
            train_idx, test_idx = d.split_data(args.fold)
            train_d = Subset(d,train_idx)
            test_d = Subset(d,test_idx)

            trainloaders[case] = DataLoader(train_d, batch_size=args.batch_size, shuffle=True)
            testloaders[case] = DataLoader(test_d, batch_size=args.batch_size, shuffle=True)
            loss_fns[case] = nn.CrossEntropyLoss()
            decoders[case] = eval(f'head{args.head}().double()')
    
    # Create model
    model = HPSModel(eval(f'encoder{args.encoder}().double()'),
                    decoders,
                    loss_fns)
    
    # Create optimizer & trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(optimizer,num_epochs=args.num_epochs,log_dir=args.log_dir)

    # Train model
    trainer.fit(model,trainloaders,testloaders)

    # Evaluate at end
    metrics = model.score(testloaders)
    for key in metrics.keys():
        print()
        print(key)
        print('Accuracy: ', metrics[key]['accuracy'])
        print('Loss: ', metrics[key]['loss'])
    print()