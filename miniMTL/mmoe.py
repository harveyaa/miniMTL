import torch
import torch.nn as nn
from miniMTL.models import *

from torcheval.metrics.functional import binary_f1_score
from torcheval.metrics.functional import binary_auroc
from torcheval.metrics.functional import binary_precision
from torcheval.metrics.functional.classification import binary_recall

class MMOEModel(nn.Module):
    """ Naive implementation of Multigate Mixture of Experts."""
    def __init__(self, expert, n_experts, decoders, loss_fns, dim_in=2080):
        super().__init__()
        self.n_experts = n_experts
        self.decoders = decoders
        self.loss_fns = loss_fns

        # create experts
        self.experts = []
        for i in range(self.n_experts):
            self.experts.append(eval(f'encoder{expert}().double()'))

        # create gates
        self.gates = {}
        for key in decoders.keys():
            self.gates[key] = nn.Linear(dim_in, n_experts).double()

        # register all the modules
        for i in range(n_experts):
            self.add_module(f'exp{i}',self.experts[i])
        for key in list(self.gates.keys()):
            self.add_module(f'gate_{key}',self.gates[key])
        for key in list(self.decoders.keys()):
            self.add_module(key,self.decoders[key])

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
        # Pass input to each expert
        X_out = []
        for expert in self.experts:
            X_out.append(expert(X))
        
        # Pass input to the gate
        soft = nn.Softmax()
        X_gate = {}
        for task in task_names:
            X_g = self.gates[task](X)
            X_gate[task] = soft(X_g)

        # Weight and recombine
        outputs = {}
        for task in task_names:
            X_ = []
            for i in range(self.n_experts):
                g = torch.unsqueeze(X_gate[task][:,i],1)
                X_.append(torch.mul(X_out[i], g))
            outputs[task] = torch.stack(X_).sum(dim=0)

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

                    if pred.size(dim=1)==2:
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