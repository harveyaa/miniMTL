import pandas as pd
import numpy as np
import os

from itertools import combinations
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir",help="path to log dir",dest='log_dir')
    parser.add_argument("--prefix",help="prefix to filter relevant run.",dest='prefix',default=None)
    parser.add_argument("--path_out",help="path to output directory",dest='path_out',default=None)
    parser.add_argument("--n_tasks",help="number of grouped tasks",dest='n_tasks',type=int)
    args = parser.parse_args()

    all_cases = ['SZ',
            'ASD',
            'BIP',
            'DEL22q11_2',
            'DUP22q11_2',
            'DEL16p11_2',
            'DUP16p11_2',
            'DEL1q21_1',
            'DUP1q21_1']

    # Get relevant log filenames
    if not args.prefix is None:
        logs = [l for l in os.listdir(args.log_dir) if l[:len(args.prefix)]==args.prefix]
    else:
        logs = os.listdir(args.log_dir)
    
    path_out = args.path_out if args.path_out is not None else args.log_dir

    print(f'Summarizing logs from:\n{args.log_dir}')
    print(f"Output to:\n{path_out}\n")

    #######################
    # LOAD SINGLE RESULTS #
    #######################
    print('Loading single task results...')
    singles = []
    single_results = []
    for case in all_cases:
        for l in logs:
            seen = l.split('.')[0].split('-')[2:]
            if (case in seen) & (len(seen)==1):
                single_results.append(pd.read_csv(os.path.join(args.log_dir,l),header=[0,1],index_col=0))
                singles.append(case)
                print(case)
    print('Done!\n')

    ######################
    # LOAD GROUP RESULTS #
    ######################
    print('Loading grouped task results...')
    groups = []
    group_results = []
    for combo in combinations(singles,args.n_tasks):
        for l in logs:
            seen = l.split('.')[0].split('-')[2:]
            if len(set(combo).intersection(set(seen))) == args.n_tasks:
                groups.append(combo)
                group_results.append(pd.read_csv(os.path.join(args.log_dir,l),header=[0,1],index_col=0))
                print(f"{' '.join(combo)}")
    print('Done!\n')
    
    ################
    # GET BASELINE #
    ################
    # Call baseline avg of best & final accuracy
    baseline = 0.5*(pd.concat(single_results,axis=1).iloc[-1] + pd.concat(single_results,axis=1).max()).loc[:,'Accuracy/test']
    print('Baseline\n--------')
    print(baseline,'\n')

    ###################
    # EVALUATE GROUPS #
    ###################
    name_to_idx = dict(zip(singles,list(range(len(singles)))))
    counts = np.zeros((len(singles),2))

    for group, results in zip(groups,group_results):
        # Increment seen
        for c in group:
            counts[name_to_idx[c],1] += 1

            val = (results.max().loc[c].loc['Accuracy/test'] + results.iloc[-1].loc[c].loc['Accuracy/test'])/2

            if val > baseline.loc[c]:
                counts[name_to_idx[c],0] += 1
    
    ####################
    # SUMMARIZE COUNTS #
    ####################
    summary_counts = pd.DataFrame(counts,index=singles,columns=['n_beat','n_seen'])
    summary_counts = summary_counts.append(summary_counts.sum().rename('Total'))
    print('Counts\n------')
    print(summary_counts)
    print()

    # Save
    filename = args.prefix + '-counts.csv' if args.prefix is not None else 'counts.csv'
    summary_counts.to_csv(os.path.join(path_out, filename))