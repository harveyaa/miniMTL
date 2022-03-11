import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_batches(dataloaders, shuffle=True):
    """
    Combine batches across DataLoaders for Multi-Task training, in either shuffled or sequential order.

    Parameters
    ----------
    dataloaders: dict[str, DataLoader]
        Dictionary from task name to DataLoader.
    shuffle: bool
        Wether or not to shuffle the order of batches.
    
    Yields
    ------
    (batch, DataLoader)
        batch is a tuple of (X, Y_dict) and DataLoader is the DataLoader
        that that batch came from - only provides metadata & is not accessed by the model.
    """
    tasks = list(dataloaders.keys())
    idx_to_task = dict(zip(range(len(tasks)),tasks))
    batch_counts = [len(dataloaders[t]) for t in tasks]

    dl_iters = [iter(dataloaders[t]) for t in tasks]

    dl_indices = []
    for idx, count in enumerate(batch_counts):
        dl_indices.extend([idx]*count)
    
    if shuffle:
        random.shuffle(dl_indices)
    
    for idx in dl_indices:
        yield next(dl_iters[idx]), dataloaders[idx_to_task[idx]]


class Trainer:
    def __init__(self,optimizer,n_epochs,log_dir='./runs/'):
        """
        Parameters
        ----------
        optimizer: Optimizer
            torch Optimizer object.
        n_epochs: int
            Number of epochs to train for.
        log_dir: str, default='./runs/'
            Path to log directory.
        """
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.writer = SummaryWriter(log_dir=log_dir)

    def fit(self,model,dataloaders,test_dataloaders):
        """
        Trains the passed model according to the hyperparameters passed to the Trainer.

        Parameters
        ----------
        model: HPSModel
            Hard Parameter Sharing model.
        dataloaders: dict[str,DataLoader]
            Dictionary from task name to DataLoader.
        """
        # Calculate the total number of batches per epoch
        tasks = list(dataloaders.keys())
        n_batches_per_epoch = np.sum([len(dataloaders[t]) for t in tasks])

        # Set to training mode
        model.train()

        for epoch_num in range(self.n_epochs):
            batches = tqdm(enumerate(get_batches(dataloaders)),
                            total=n_batches_per_epoch,
                            desc=f"Epoch {epoch_num}")
            for batch_num, (batch, dataloader) in batches:
                X, Y_dict = batch
                batch_size = len(next(iter(Y_dict.values())))

                # Set gradients of all model parameters to zero
                self.optimizer.zero_grad()

                # Perform forward pass and calcualte the loss and count
                loss_dict = model.calculate_loss(X, Y_dict)

                # Update log
                for task in tasks:
                    if task in loss_dict.keys():
                        self.writer.add_scalar(f"Loss/train/{task}",loss_dict[task],epoch_num)

                # Calculate the average loss
                if len(loss_dict.values()) == 1:
                    loss = loss_dict[list(loss_dict.keys())[0]]

                # Perform backward pass to calculate gradients
                loss.backward()

                # Clip gradient norm

                # Update the parameters
                self.optimizer.step()

                # Evaluate model
                metrics = model.score(test_dataloaders)
                for task in tasks:
                    self.writer.add_scalar(f"Loss/test/{task}",metrics[task]['test_loss'],epoch_num)
                    self.writer.add_scalar(f"Accuracy/test/{task}",metrics[task]['accuracy'],epoch_num)

        self.writer.flush()
        self.writer.close()