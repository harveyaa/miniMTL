import numpy as np
import random

def get_batches(dataloaders, shuffle=True):
    """
    Parameters
    ----------
    dataloaders: dict str: dataloader
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
    def __init__(self,optimizer,n_epochs):
        self.optimizer = optimizer
        self.n_epochs = n_epochs

    def fit(self,model,dataloaders):
        # Calculate the total number of batches per epoch
        tasks = list(dataloaders.keys())
        n_batches_per_epoch = np.sum([len(dataloaders[t]) for t in tasks])

        # Set training helpers/logging

        # Set to training mode
        model.train()

        for epoch_num in range(self.n_epochs):
            batches = enumerate(get_batches(dataloaders))
            for batch_num, (batch, dataloader) in batches:
                X, Y_dict = batch

                #total_batch_num = epoch_num * self.n_batches_per_epoch + batch_num
                #batch_size = len(next(iter(Y_dict.values())))

                # Set gradients of all model parameters to zero
                self.optimizer.zero_grad()

                # Perform forward pass and calcualte the loss and count
                loss_dict = model.calculate_loss(X, Y_dict)

                # Update running loss and count

                # Calculate the average loss
                if len(loss_dict.values()) == 1:
                    loss = loss_dict[list(loss_dict.keys())[0]]

                # Perform backward pass to calculate gradients
                loss.backward()

                # Clip gradient norm
                #if self.config.grad_clip:
                #    torch.nn.utils.clip_grad_norm_(
                #        model.parameters(), self.config.grad_clip
                #    )

                # Update the parameters
                self.optimizer.step()