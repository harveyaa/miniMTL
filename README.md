# miniMTL

Minimal Mutli-Task Learning for neuroimaging data.

```
├── README.md
│
├── setup.py
│
├── LICENSE
│
├── requirements.txt            <- generated with `pip freeze > requirements.txt`
│
├── miniMTL                     <- main code base
│   ├── __init__.py
│   │
│   ├── datasets.py             <- classes to generate torch Datasets
│   │
│   ├── hps.py                  <- class to build a Hard Parameter Sharing model (HPSModel) from component models (encoder & decoders)
│   │
│   ├── logging.py              <- class for simple logging of training data to csv
│   │
│   ├── models.py               <- home for component models
│   │
│   ├── training.py             <- class for training a HPSModel
│   │
│   └── util.py                 <- util functions
│
├── tests                       <- pytest tests
│   └── tests.py
│
├── examples                    <- scripts to run models
│   ├── model_00.py             <- HPS with encoder0 and head0
│   │
│   ├── ukbb_sex.py             <- relic (for now) single task using ukbb sex prediction
│   │
│   └── arch_search             <- scripts to facilitate model evaluation
│       ├── compare.ipynb       <- draft notebook (contains plotting functions)
│       │
│       ├── summarize_pairs.py  <- combines logs for pairwise task model evaluation
│       │
│       └── local.sh            <- runs local pipeline for pairwise task model evaluation
│
└── workbook.ipynb              <- draft notebook
```

## TODO:
- implement:
  - learning rate scheduling
  - clip gradient norms
  - other features/hyperparameters
- use train/test/val splits
  - need more reliable baseline values for small CNV groups
- make better use of UKBB data
  - sex prediction as auxiliary task
  - other auxiliary tasks (confounds?)
  - build imbalanced datasets for CNVs?
    - makes metrics messier (accuracy baseline no longer 50)
- How to evalute HPS model beyond pairwise
  - Want a model that overall improves performance for each task vs single task
  - groups of 3+ vs baseline?
    - subsample from possible groupings
