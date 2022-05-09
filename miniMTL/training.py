import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from miniMTL.logging import Logger

def get_batches(dataloaders, shuffle=True,seed=0):
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
    batch
        batch is a tuple of (X, Y_dict)
    """
    tasks = list(dataloaders.keys())
    batch_counts = [len(dataloaders[t]) for t in tasks]

    dl_iters = [iter(dataloaders[t]) for t in tasks]

    dl_indices = []
    for idx, count in enumerate(batch_counts):
        dl_indices.extend([idx]*count)
    
    if shuffle:
        # TODO: seed or not?
        # Do we want same batch order across eopchs? (guess no)
        #random.seed(seed)
        random.shuffle(dl_indices)
    
    for idx in dl_indices:
        yield next(dl_iters[idx])


class Trainer:
    def __init__(self,optimizer,lr_scheduler=None,num_epochs=100,clip_grad=True,max_grad=1,log_dir=None):
        """
        Parameters
        ----------
        optimizer: Optimizer
            torch Optimizer object.
        lr_scheduler: LRScheduler, default=None
            torch LRScheduler object, initialized with corresponding optimizer.
        num_epochs: int
            Number of epochs to train for.
        clip_grad: bool, default=True
            Wether or not to clip gradient norm.
        max_grad: float, default=1
            Maximum gradient to clip.
        log_dir: str, default=None
            Path to log directory.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.clip_grad = clip_grad
        self.max_grad = max_grad
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger = Logger(log_dir=log_dir)

    def fit(self,model,dataloaders,test_dataloaders, shuffle=True):
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

        for epoch_num in range(self.num_epochs):
            batches = tqdm(enumerate(get_batches(dataloaders,shuffle=shuffle)),
                            total=n_batches_per_epoch,
                            desc=f"Epoch {epoch_num}")
            for batch_num, batch in batches:
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
                        self.logger.add_scalar(task,"Loss/train",loss_dict[task].item(),epoch_num)

                # Calculate the average loss
                if len(loss_dict.values()) == 1:
                    loss = loss_dict[list(loss_dict.keys())[0]]

                # Perform backward pass to calculate gradients
                loss.backward()

                # Clip gradient norm
                if self.clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad)

                # Update the parameters
                self.optimizer.step()

                # Evaluate model
                metrics = model.score(test_dataloaders)
                for task in tasks:
                    self.writer.add_scalar(f"Loss/test/{task}",metrics[task]['loss'],epoch_num)
                    self.writer.add_scalar(f"Accuracy/test/{task}",metrics[task]['accuracy'],epoch_num)
                    
                    self.logger.add_scalar(task, "Loss/test",metrics[task]['loss'],epoch_num)
                    self.logger.add_scalar(task, "Accuracy/test",metrics[task]['accuracy'],epoch_num)

            self.lr_scheduler.step()

        self.writer.flush()
        self.writer.close()
        self.logger.save()