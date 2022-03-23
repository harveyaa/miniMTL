
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def split_data(dataset,test_size=0.2,seed=0):
    train_idx, test_idx, _, _ = train_test_split(range(len(dataset)),
                                                dataset.Y,
                                                stratify=dataset.Y,
                                                test_size=test_size,
                                                random_state=seed)

    return Subset(dataset,train_idx), Subset(dataset,test_idx)