import os
import pandas as pd
from datetime import datetime

class Logger:
    """ Extra simple logging class to save a .csv of metrics per epoch per task.

    Attributes
    ----------
    log_dir: str
        Log directory, if None passed creates a dir 'logs' in current location.
    tasks: dict[str:dict[str:dict[int:float]]]
        Dictionary from task to dictionary of metric labels, to epoch:value.
    filename: str
    """
    def __init__(self,log_dir=None):
        self.log_dir = log_dir if not log_dir is None else ''

        if not os.path.isdir(self.log_dir):
            default_dir = os.path.join(os.getcwd(),'logs')
            if not os.path.isdir(default_dir):
                os.mkdir(default_dir)
            self.log_dir = default_dir

        self.tasks = {}

    def add_scalar(self,task, label, val, epoch):
        if task not in self.tasks.keys():
            self.tasks[task] = {}
        if not label in self.tasks[task].keys():
            self.tasks[task][label] = {}
        self.tasks[task][label][epoch] = val
    
    def save(self):
        dfs = []
        for task in list(self.tasks.keys()):
            dfs.append(pd.DataFrame(self.tasks[task]))
        df = pd.concat(dfs,keys = list(self.tasks.keys()),axis=1)
        
        self.filename = datetime.now().strftime("%Y:%m:%d-%H:%M:%S-") + '-'.join(list(self.tasks.keys())) + '.csv'
        df.to_csv(os.path.join(self.log_dir,self.filename))