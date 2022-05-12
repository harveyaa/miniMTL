import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def split_data(dataset,splits=(0.8,0.2),seed=None):
    assert np.sum(splits) == 1
    assert (len(splits)==2) | (len(splits)==3)

    if len(splits) == 2:
        train_idx, test_idx, _, _ = train_test_split(range(len(dataset)),
                                                    dataset.Y,
                                                    stratify=dataset.Y,
                                                    test_size=splits[1],
                                                    random_state=seed)

        return Subset(dataset,train_idx), Subset(dataset,test_idx)
    else:
        train_idx, test_idx, _, _ = train_test_split(range(len(dataset)),
                                                    dataset.Y,
                                                    stratify=dataset.Y,
                                                    test_size=(splits[1]+splits[2]),
                                                    random_state=seed)
        
        test_idx, val_idx, _, _ = train_test_split(test_idx,
                                                    dataset.Y[test_idx],
                                                    stratify=dataset.Y[test_idx],
                                                    test_size=splits[2]/(splits[1] + splits[2]),
                                                    random_state=seed)
        return Subset(dataset,train_idx), Subset(dataset,test_idx), Subset(dataset,val_idx)