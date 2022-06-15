# Directed Graph rewiring code

Ilias Rentzeperis, Steeve Laquitaine

Please cite the following paper:

```
@article{rentzeperis2022adaptive,
  title={Adaptive rewiring of random neural networks generates convergent--divergent​ units},
  author={Rentzeperis, Ilias and Laquitaine, Steeve and van Leeuwen, Cees},
  journal={Communications in Nonlinear Science and Numerical Simulation},
  volume={107},
  pages={106135},
  year={2022},
  publisher={Elsevier}
}
```

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

## Run 

1. Create raw directed graphs data (can take up to 2 days)

Open and run all cells of `1RunStoreDigraphs.ipynb`. After re-creating
the raw data you can either:  
   1. execute the analyses and plot the figures (1)
   2. or directly plot the figures from the pre-stored analyses results.

1. Run analyses on raw data and create figures: 

```bash
python -m main --run figure2
python -m main --run figure3
python -m main --run figure4
python -m main --run figure5
python -m main --run figure6
python -m main --run figureS3
python -m main --run figureS4
```

3. Or just load stored intermediate analyses and quickly re-create figures:

```bash
python -m main --load figure2
python -m main --load figure3
python -m main --load figure4
python -m main --load figure5
python -m main --load figure6
python -m main --load figureS3
python -m main --load figureS4
```
