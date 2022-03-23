import torch
import torch.nn as nn

class HPSModel(nn.Module):
    """ Multi-input HPS."""
    def __init__(self,encoder,decoders,loss_fns):
        """
        Parameters
        ----------
        encoder: nn.Module
            Shared portion of the model.
        decoders: dict[str, nn.Module]
            Dictionary from task name to task-specific decoder (head).
        loss_fns: dict[str, loss_fn]
            Dictionary from task name to torch loss function.
        """
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.loss_fns = loss_fns

        # Register all the modules to make sure the parameters are tracked properly
        # If don't do this explicitly model.parameters() are only for encoder
        self.add_module('encoder',encoder)
        for key in list(self.decoders.keys()):
            self.add_module(key,decoders[key])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        print(f'Initialized HPSModel using: {self.device}.\n')
    
    def forward(self,X,task_names):
        """
        Parameters
        ----------
        X: Tensor
            A batch of data.
        task_names: list[str]
            List of task names associated with X.

        Returns
        -------
        dict[str, Tensor]
            Dictionary from task name to associated output.
        """
        X = self.encoder(X)

        outputs = {}
        for task in task_names:
            outputs[task] = self.decoders[task](X)
        return outputs
    
    def calculate_loss(self,X,Y_dict):
        """
        Parameters
        ----------
        X: Tensor
            A batch of data.
        Y_dict: dict[str, Tensor]
            Dictionary from task name to associated labels.
        
        Returns
        -------
        dict[str, Tensor]
            Dictionary from task name to associated loss.
        """
        task_names = Y_dict.keys()
        outputs = self.forward(X.to(self.device),task_names)
        losses = {}
        for task in task_names:
            losses[task] = self.loss_fns[task](outputs[task],Y_dict[task].to(self.device))
        return losses
    
    def score(self,dataloaders):
        """ 
        Score model on given data.

        Parameters
        ----------
        dataloaders: dict[str, DataLoader]
            Dictionary from task name to associated DataLoader.
        
        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary from task name to dictionary of metric label
            ('accuracy', 'test_loss') to value.
        """
        tasks = list(dataloaders.keys())
        self.eval()
        
        metrics = {}
        for task in tasks:
            dataloader = dataloaders[task]
            loss_fn = self.loss_fns[task]
            size = len(dataloader.dataset)

            test_loss, correct = 0, 0
            with torch.no_grad():
                i=0
                for X, Y_dict in dataloader:
                    X = X.to(self.device)                    
                    Y = Y_dict[task].to(self.device)
                    pred = self.forward(X,[task])[task]

                    test_loss += loss_fn(pred, Y).item()
                    correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
                    i +=1
            test_loss /= size
            correct /= size
            metrics[task] = {'accuracy':100*correct,'test_loss':test_loss}
        return metrics