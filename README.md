# Geodesic Graph Neural Network for Efficient Graph Representation Learning

CS 768 Learning with Graphs project. Harsh Poonia (210050063) and Kartik Nair (210050083)

1. First Checkpoint Presentation - slides [here](https://docs.google.com/presentation/d/1BJGQ3L3FJOUIr1rvl-3YU9ci1oHyK-DqjT34xNfdFs4/edit?usp=sharing). An introduction to the paper, understanding the methods and architecture employed
2. Second Checkpoint Presentation - slides [here](https://docs.google.com/presentation/d/1hgPYwDda8gny4fnEI5Ffr8DmbC_SHBzTotOL0CXjs6E/edit?usp=sharing). Reproducibility study on the paper, and our results, contrasted with those obtained by the authors. Our analysis of our obtained results
3. Final Presentation - slides [here](https://docs.google.com/presentation/d/1WHMwS0_NyacciubnpBLKZ11bTeP3d6EsMQ16YSymtMs/edit?usp=sharing). Our modifications to the code, to implement the extensions we proposed in the first presentation. Results obtained from these, and our analyses thereof.


## Installation

We recommend installation from Conda:

```bash
git clone https://github.com/woodcutter1998/gdgnn.git
cd gdgnn
sh setup.sh
```

## Usage

`--gd_type` controls the type of geodesics, it can either be `VerGD` for vertical geodesics or `HorGD` for horizontal geodesics.

`--num_layers` controls the number of layers in the GNN.

To run different datasets, do `python run_**.py` with the parameters specified.

To search for hyperparameters, modify the `hparams` variable in the `run_**.py` files to specify the list of potentail hyperparameters. e.g.
```python
hparams = {'num_layers':{2,3,4,5},
            'gd_type':{'VerGD, HorGD'},
            'dropout':{0.5, 0.7, 0.9}}
``` 
and do `python run_**.py --psearch True`. The program performs grid-search on the hyperparameters specified.

An example command to reproduce our results on OGBG-MOLHIV dataset is:

```bash
python run_MOL.py --train_data_set ogbg-molhiv --psearch True
```
