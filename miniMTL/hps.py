import torch
import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score
from torcheval.metrics.functional import binary_auroc
from torcheval.metrics.functional import binary_precision
from torcheval.metrics.functional.classification import binary_recall

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

            loss, correct, auc, f1, precision, recall = 0, 0, 0, 0, 0, 0
            with torch.no_grad():
                for X, Y_dict in dataloader:
                    X = X.to(self.device)                    
                    Y = Y_dict[task].to(self.device)
                    pred = self.forward(X,[task])[task]

                    # Do this to average properly over the whole list of batches
                    loss += loss_fn(pred, Y).item()*X.size(dim=0)

                    probs = nn.functional.softmax(pred,dim=1)[:,1]
                    auc += binary_auroc(probs,Y).item()

                    pred_labels = pred.argmax(1)
                    correct += (pred_labels == Y).type(torch.float).sum().item()

                    f1 += binary_f1_score(probs,Y).item()
                    precision += binary_precision(probs,Y).item()
                    recall += binary_recall(probs,Y).item()
            loss /= size
            correct /= size
            auc /= len(dataloader)
            f1 /= len(dataloader)
            precision /= len(dataloader)
            recall /= len(dataloader)

            metrics[task] = {'accuracy':100*correct,
                             'loss':loss,
                             'auc':auc,
                             'f1':f1,
                             'precision':precision,
                             'recall':recall}
        return metrics