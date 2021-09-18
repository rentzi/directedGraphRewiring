# Directed Graph rewiring

author: Ilias Rentzeperis, Steeve Laquitaine, Cees van leeuwen


## Setup

```bash
conda create -n dgr python==3.7      # create dgr virtual environment  
pip install -r src/requirements.txt  # install requirements.txt 
ipython kernel install --name dgr    # create jupyter kernel for dgr
```

## run 

Run analysis on raw data and create figure: 

```bash
python -m main --run figure2
```

Load intermediate analysis and plot figure:

```bash
python -m main --load figure2
```