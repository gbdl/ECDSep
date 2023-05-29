# Improving Energy Conserving Descent for Machine Learning: Theory and Practice
In this repository we collect the implementation of the optimizer ECDSep described in *Improving Energy Conserving Descent for Machine Learning: 
Theory and Practice* together with the code to reproduce the experiments presented in the paper.

## Requirements
The requirements are in the file `requirements.txt`.

## Importing the optimizer
The script `inflation.py` contains the code of ECDSep. To import the optimizer in your Python code it suffices to put the file in the folder of your code
and add
```
from inflation import ECDSep
```
at the beginning of it. To call it write
```
ECDSep(parameters, lr, nu=1e-5, eta, weight_decay=0, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=0., s=1, consEn=True)
```
where
- `parameters` are the parameters regarding which performing optimization.
- `lr` is the learning rate value and is required.
- `nu` the chaos hyperparameter $\nu$.
- `eta` the measure concentration hyperparameter $\eta$ (required)
		It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
- `F0` is the $F_0$ value.
- `eps1` and `eps2` are two constants that improve stability.
- `deltaEn` is the $\delta E$ value.
- `s` is the regularization switch $s$.
- `weight_decay` is the weight decay value.
- `consEn` indicates whether energy is conserved or not during optimization.

## Reproducing the experiments
The paper contains several experiments comparing ECDSep against Adam, AdamW and SGD.


### Synthetic
The experiments on the Ackley and Zakharov functions are in the notebooks `ECDSep_ackley.ipynb` and `ECDSep_zakharov.ipynb` respectively. To reproduce them just execute the cells in the notebooks (note that these exepriments do not require a GPU). The hyperparameters that give the best results are searched using [optuna](https://optuna.readthedocs.io/en/stable/) for all optimizers. 


### Images

#### CIFAR 100
The folder contains the code for performing the CIFAR-100 SWA experiments using the SWA capabilities of Pytorch. This has been adapted from [https://github.com/izmailovpavel/torch_swa_examples](https://github.com/izmailovpavel/torch_swa_examples), to which Adam, AdamW and ECDSep have been added.

To run the CIFAR100 experiments, move to the folder and then run `train.py` as follows.

For SGD:
```
cd cifar-swa
python3 train.py --dir=cifar_output --dataset=CIFAR100 --data_path=cifar100 --model=WideResNet28x10 --epochs=300 --lr_init=0.05 --swa_lr=0.05 --wd=5e-4 --swa --swa_start=161 --optimizer=sgd --momentum=0.9 --seed=42 
```

For ECDSep:

```
cd cifar-swa
python3 train.py --dir=cifar_output --dataset=CIFAR100 --data_path=cifar100 --model=WideResNet28x10 --epochs=300 --lr_init=0.4 --wd=5e-4 --swa --swa_start=161  --nu=5e-5 --eta=1.0 --deltaEn=0.0 --F0=0.0 --optimizer=ECDSep --seed=42 
```

For Adam:

```
cd cifar-swa
python3 train.py --dir=cifar_output --dataset=CIFAR100 --data_path=cifar100 --model=WideResNet28x10 --epochs=300 --lr_init=0.0001 --wd=1e-4 --swa --swa_start=161 --swa_lr=0.0001 --optimizer=adam --seed=42
```

For AdamW:

```
cd cifar-swa
python3 train.py --dir=cifar_output --dataset=CIFAR100 --data_path=cifar100 --model=WideResNet28x10 --epochs=300 --lr_init=0.0001 --wd=1e-4 --swa --swa_start=161 --swa_lr=0.0001 --optimizer=adamw --seed=42
```

The hyperparameters that give the best performance (accuracy) are:
- ECDSep: $\nu = 5\times 10^{-5}$, $\Delta t = 0.4$
- SGD: $\alpha = 0.05$, $\beta = 0.9$.
- Adam: $\alpha = 10^{-4}$, $w_d = 10^{-4}$
- AdamW: $\alpha = 10^{-4}$, $w_d = 10^{-4}$



#### Imagenet and Tiny Imagenet

For these experiments, the SWA procedure is not implemented during training, but the networks at each epoch are saved and averaged at the end. This gives more flexibility to choose the best epoch at which to start the average.

The code for running this experiments is in the subfolder `ECDSep-IN`, together with the notebooks to perform this analysis.
For tiny-IN the code will automatically download the dataset, but for IN this has to been provided by the user. We thank [Daniel Kunin](https://github.com/danielkunin) for sharing this code.


##### Tiny-Imagenet

The code to run the experiments for each optimizer is below.

SGD
```
cd ECDSep-IN
python3 train.py --experiment run-SGD --expid 1 --model-class=tinyimagenet --model resnet18 --dataset tiny-imagenet --wd 1e-4 --train-batch-size 128 --test-batch-size 128 --gpu 0 --lr 0.1 --epochs 100 --optimizer momentum --momentum 0.9 --seed 42 --overwrite 
```

ECDSep
```
cd ECDSep-IN
python3 train.py --experiment run-ECD --expid 1 --model-class=tinyimagenet --model resnet18 --dataset tiny-imagenet --wd 1e-4 --nu 5e-5 --train-batch-size 128 --test-batch-size 128 --gpu 0 --lr 0.6 --epochs 100 --eta 1.0 --F0 0.0 --deltaEn 0.0 --optimizer ECDSep --seed 42 --overwrite 
```

Adam
```
cd ECDSep-IN
python3 train.py --experiment run-adam --expid 1 --model-class=tinyimagenet --model resnet18 --dataset tiny-imagenet --wd 1e-4  --lr 0.001 --train-batch-size 128 --test-batch-size 128 --gpu 0 --epochs 100 --optimizer adam  --seed 42 --overwrite
```

For AdamW
```
cd ECDSep-IN
python3 train.py --experiment run-adamw --expid 1 --model-class=tinyimagenet --model resnet18 --dataset tiny-imagenet --wd 1e-4  --lr 0.001 --train-batch-size 128 --test-batch-size 128 --gpu 0 --epochs 100 --optimizer adamw  --seed 42 --overwrite
```

The parameters that give the best performance (accuracy) for each optimizer are:

- ECDSep: $\Delta t = 0.6, \nu = 5\times 10^{-5}$.
- SGD: $\alpha = 0.1, \beta = 0.9$.
- Adam: $\alpha = 10^{-3}, w_d = 10^{-4}$.
- AdamW: $\alpha = 10^{-3}, w_d = 10^{-4}$.

After the run is done, the notebook `notebooks/averages.ipynb` can be used to average the saved networks over an arbitrary number of epochs (starting from the last one) and keep the averaged network with the best test accuracy.


##### Imagenet 1-K
For Imagenet-1K, the dataset has to be provided. In the commands below replace `DATADIR` with the directory where the IN-1K dataset is located.

For SGD
```
cd ECDSep-IN
python3 train.py --experiment run-SGD --expid 1 --model-class=imagenet --model resnet18 --pretrained True --dataset imagenet --data-dir DATADIR --wd 1e-4 --train-batch-size 128 --test-batch-size 128 --gpu 0 --lr 5e-5 --epochs 10 --momentum 0.99 --optimizer momentum --seed 42 --overwrite
```

For ECDSep
```
cd ECDSep-IN
python3 train.py --experiment run-ECD --expid 1 --model-class=imagenet --model resnet18 --pretrained True --dataset imagenet --data-dir DATADIR --wd 1e-4 --nu 1e-3 --train-batch-size 128 --test-batch-size 128 --gpu 0 --lr 0.05 --epochs 10 --eta 1.0 --F0 0.0 --deltaEn 0.0 --optimizer ECDSep --seed 42 --overwrite
```

For Adam
```
cd ECDSep-IN
python3 train.py --experiment run-adam --expid 1 --model-class=imagenet --model resnet18 --pretrained True --dataset imagenet --data-dir DATADIR --wd 1e-4 --train-batch-size 128 --test-batch-size 128 --gpu 0 --lr 1e-5 --epoch&s 10 --optimizer adam --seed 42 --overwrite
```

For AdamW
```
cd ECDSep-IN
python3 train.py --experiment run-adamw --expid 1 --model-class=imagenet --model resnet18 --pretrained True --dataset imagenet --data-dir DATADIR --wd 1e-4 --train-batch-size 128 --test-batch-size 128 --gpu 0 --lr 1e-5 --epochs 10 --optimizer adamw --seed 42 --overwrite
```

After the run is done, the notebook `notebooks/averagesIN.ipynb` can be used to average the saved networks over an arbitrary number of epochs (starting from the last one) and keep the averaged network with the best test accuracy.


The hyperparameters that give the best performance (accuracy) are:
- ECDSep: $\Delta t = 0.1$ , $\nu = 10^{-3}$, $w_d = 10^{-4}$.
- SGD: $\alpha = 5\times 10^{-5}$, $\beta = 0.99$, $w_d=10^{-4}$.
- Adam: $\alpha =  10^{-5}$, $w_d = 10^{-4}$.
- AdamW: $\alpha =  10^{-5}$, $w_d = 10^{-4}$.

Summing up, the best accuracy results of all optimizers are summarized in the following table. The results are averaged over $4$, $3$ and $2$ runs for CIFAR100, Tiny Imagenet and Imagenet-1K respectively.


|                     |  ECDSep  |  SGD  |  Adam  |  AdamW  |
|---------------------|----------|-------|--------|---------|
| CIFAR 100           |  82.57   | 82.50 | 79.01  |  78.71  | 
| Tiny Imagenet       |  66.44   | 64.83 | 61.67  |  59.84  |
| IN-1K (fine tuning) |  70.49   | 70.49 | 70.48  |  70.48  |


### Graphs

The experiments on graphs are collected in the Colab notebooks `ECDSep_graphs_arxiv.ipynb` and `ECDSep_graphs_proteins.ipynb`. To reproduce them just execute the cells in the notebooks changing the optimizers and the hyperparameters. Note that the required packages are installed automatically when executing the notebooks. The code for the datasets and the training is taken from the [Open Graph Benchmark repository](https://github.com/snap-stanford/ogb/tree/master/examples) and is released under MIT License. The hyperparameters that give the best results for each optimizer are:

- For `ogbn-arxiv`
    - ECDSep: $\Delta t=2.8,\ \eta=4.5,\ \nu=10^{-5},\ w_d=0$ (and the default remaining hyperparameters).
    - SGD: $\alpha=0.1$, $\beta=0.95$, $w_d=10^{-3}$.
    - AdamW: $\alpha=5\cdot 10^{-3}$, $w_d=0$.
    - Adam: $\alpha=5\cdot 10^{-3}$, $w_d=0$.
- For `ogbn-proteins`
    - ECDSep: $\Delta t=1.8,\ \eta=5,\ \nu=10^{-5},\ w_d=0$ (and the default remaining hyperparameters).
    - SGD: $\alpha=0.1$, $\beta=0.999$, $w_d=10^{-5}$.
    - AdamW: $\alpha=0.01$, $w_d=10^{-5}$.
    - Adam: $\alpha=0.01$, $w_d=0$.

The best performance (accuracy for `ogbn-arxiv` and ROC-AUC score for `ogbn-proteins`) of all optimizers is summarized in the following table. For `ogbn-arxiv` the results are averaged over $10$ runs, while for `ogbn-proteins` over $5$ runs.

|                  |  ECDSEp  |  SGD  |  Adam  |  AdamW  |
| ---------------- |----------|-------|--------|---------|
| `ogbn-arxiv`     |  71.55   | 71.81 |  72.37 |  72.41  |
| `ogbn-proteins`  |  74.67   | 65.79 |  77.42 | 77.44   |

### Language
The NLP exepriments on the [GLUE benchmark](https://gluebenchmark.com/) are contained in the script `ECDSep_language_bert.py` and are perfomed using the [Hugging Face](https://huggingface.co/) tools. The To reproduce them just run
```
python3 ECDSep_language_bert.py
```
with the following arguments:
- ``--optimizer`` : the optimizer chosen to perform the experiments among ECDSep, adam, adamw and sgd. Default is ECDSep.
- ``--lr``: the value of the learning rate. Default is $0.04$.
- ``--momentum``: the momentum value for SGD. Default is $0.99$.
- ``--nu``: the $\nu$ value for ECDSep. Default is $10^{-5}$.
- ``--eta``: the $\eta$ value for ECDSep. Default is $1.4$.
- ``--consEn``: whether energy is conserved or not in ECDSep. Default is True.
- ``--F0``: the $F_0$ value for ECDSep. Default is $0$.
- ``--deltaEn``: $\delta E$ value for ECDSep. Default is $0$.
- ``--s``: regularization $s$ value for ECDSep. Default is $1$.
- ``--epochs``: number of epochs. Default is $3$.
- ``--dataset``: name of the GLUE dataset. Default "all" (meaning that the training is done over all datasets).
- ``--seed``: random seed to use in the experiment. Default is $42$.
- ``--wd``: weight decay value. Default is $0$.

The hyperparameters that give the best results for each optimizer are:
- For MNLI:
	- ECDSep: $w_d = 0$, $\eta  = 2$ , $\nu  =  10^{-4}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-5}$, $\beta  = 0.99$.
	- AdamW: $w_d = 10^{-2}$, $\alpha = 2\times 10^{-5}$.
	- Adam: $w_d = 0$, $\alpha = 2 10^{-5}$.
- For QQP:
	- ECDSep: $w_d = 0$, $\eta  = 2$ , $\nu  =  10^{-4}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-5}$, $\beta  = 0.99$.
	- AdamW: $w_d = 10^{-2}$, $\alpha = 2 10^{-5}$.
	- Adam: $w_d = 0$, $\alpha = 2 \times 10^{-5}$.
- For QNLI:
	- ECDSep: $w_d = 0$, $\eta  = 1.4$ , $\nu  =  10^{-5}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-5}$, $\beta  = 0.99$.
	- AdamW: $w_d = 10^{-2}$, $\alpha = 2\times 10^{-5}$.
	- Adam: $w_d = 0$, $\alpha = 2 \times 10^{-5}$.
- For SST-2:
	- ECDSep: $w_d = 0$, $\eta  = 1$ , $\nu  =  10^{-4}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-5}$, $\beta  = 0.99$.
	- AdamW: $w_d = 0$, $\alpha = 2 \times 10^{-5}$.
	- Adam: $w_d = 0$, $\alpha = 2\times 10^{-5}$.
- For CoLA:
	- ECDSep: $w_d = 10^{-2}$, $\eta  = 2$ , $\nu  =  10^{-5}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-4}$, $\beta  = 0.9$.
	- AdamW: $w_d = 10^{-3}$, $\alpha = 3\times 10^{-5}$.
	- Adam: $w_d = 10^{-5}$, $\alpha = 2\times 10^{-5}$.
- For STS-B:
	- ECDSep: $w_d = 10^{-2}$, $\eta  = 2$ , $\nu  =  10^{-5}$.
	- SGD: $w_d = 10^{-2}$, $\alpha = 10^{-5}$, $\beta  = 0.99$.
	- AdamW: $w_d = 0$, $\alpha = 3\times 10^{-5}$.
	- Adam : $w_d = 0$, $\alpha = 2\times 10^{-5}$.
- For MRPC:
	- ECDSep: $w_d = 10^{-3}$, $\eta  = 1.4$ , $\nu  =  10^{-5}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-4}$, $\beta  = 0.99$.
	- AdamW: $w_d = 0$, $\alpha = 2\times 10^{-5}$.
	- Adam: $w_d = 10^{-2}$, $\alpha = 2\times 10^{-5}$.
- For RTE:
	- ECDSep: $w_d = 10^{-3}$, $\eta  = 1$ , $\nu  =  10^{-5}$.
	- SGD: $w_d = 10^{-3}$, $\alpha = 10^{-4}$, $\beta  = 0.99$.
	- AdamW: $w_d = 10^{-2}$, $\alpha = 3\times 10^{-5}$.
	- Adam: $w_d = 10^{-2}$, $\alpha = 2\times 10^{-5}$.


The best performance (Matthews correlation for CoLA, Spearmans’s correlation for STS-B, F1 score for MRPC and QQP, and accuracy for the remaining datasets) of all optimizers is summarized in the following table. The results are averaged over $3$ runs.

|          |  MNLI  |  QQP  |  QNLI  |  SST-2  |  CoLA  |  STS-B  |  MRPC  |  RTE  |  avg.  |
|--------- |--------|-------|--------|---------|--------|---------|--------|-------|--------|
| ECDSep   | 84.24  | 86.70 | 91.19  | 92.66   | 57.91  |  89.26  | 90.96  | 73.16 | 83.26  |
| SGD      | 83.31  | 86.36 | 91.03  | 92.17   | 60.54  |  89.26  | 90.88  | 71.96 | 83.19  |  
| Adam     | 84.31  | 88.14 | 91.39  | 92.81   | 59.34  |  89.02  | 91.09  | 71.36 | 83.43  |
| AdamW    | 84.41  | 88.21 | 91.49  | 93.03   | 59.68  |  89.15  | 91.13  | 71.24 | 83.54  |



