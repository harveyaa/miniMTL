# miniMTL

Minimal Mutli-Task Learning for neuroimaging data.

```
├── LICENSE
│
├── setup.py
│
├── README.md
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
