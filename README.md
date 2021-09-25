# Directed Graph rewiring code

Ilias Rentzeperis, Steeve Laquitaine

## Prerequisites

* `Conda` must be installed

## Setup

Move to your project’s root directory.

```bash
conda create -n dgr python==3.7      # create dgr virtual environment  
conda activate dgr
pip install -r src/requirements.txt  # install requirements.txt 
ipython kernel install --name dgr    # create jupyter kernel for dgr
```

## run 

Run analysis on raw data and create figure: 

```bash
python -m main --run figure2
python -m main --run figure3
python -m main --run figure4
python -m main --run figure5
python -m main --run figure6
python -m main --run figureS3
python -m main --run figureS4
```

Load intermediate analysis and plot figure:

```bash
python -m main --load figure2
python -m main --load figure3
python -m main --load figure4
python -m main --load figure5
python -m main --load figure6
python -m main --load figureS3
python -m main --load figureS4
```
