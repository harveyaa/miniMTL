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

    #####################
    # LOAD PAIR RESULTS #
    #####################
    print('Loading paired task results...')
    pairs = []
    pair_results = []
    for case1,case2 in combinations(singles,2):
        for l in logs:
            seen = l.split('.')[0].split('-')[2:]
            if (case1 in seen) and (case2 in seen):
                pair_results.append(pd.read_csv(os.path.join(args.log_dir,l),header=[0,1],index_col=0))
                pairs.append((case1,case2))
                print(f"{case1} {case2}")
    print('Done!\n')
    
    ################
    # GET BASELINE #
    ################
    # Call baseline avg of best & final accuracy
    baseline = 0.5*(pd.concat(single_results,axis=1).iloc[-1] + pd.concat(single_results,axis=1).max()).loc[:,'Accuracy/test']
    print('Baseline\n--------')
    print(baseline,'\n')

    ##################
    # EVALUATE PAIRS #
    ##################
    name_to_idx = dict(zip(singles,list(range(len(singles)))))
    counts = np.zeros((len(singles),2))
    vals = np.zeros((len(singles),len(singles)))

    for pair, results in zip(pairs,pair_results):
        case1 = pair[0]
        case2 = pair[1]

        # Increment seen
        counts[name_to_idx[case1],1] += 1
        counts[name_to_idx[case2],1] += 1

        # Record vals
        case1_val = (results.max().loc[case1].loc['Accuracy/test'] + results.iloc[-1].loc[case1].loc['Accuracy/test'])/2
        case2_val = (results.max().loc[case2].loc['Accuracy/test'] + results.iloc[-1].loc[case2].loc['Accuracy/test'])/2

        # vals[case1,case2] = performance of case1 w/ case2
        vals[name_to_idx[case1],name_to_idx[case2]] = case1_val
        vals[name_to_idx[case2],name_to_idx[case1]] = case2_val

        # Increment beat    
        if case1_val > baseline.loc[case1]:
            counts[name_to_idx[case1],0] += 1
        if case2_val > baseline.loc[case2]:
            counts[name_to_idx[case2],0] += 1
    
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

    ##################
    # SUMMARIZE VALS #
    ##################
    # Make diagonal the baseline
    np.fill_diagonal(vals,baseline.values)
    summary_vals = pd.DataFrame(vals,index=singles,columns=singles)
    print('Values\n------')
    print(summary_vals)
    print()

    # Save
    filename = args.prefix + '-vals.csv' if args.prefix is not None else 'vals.csv'
    summary_vals.to_csv(os.path.join(path_out, filename))