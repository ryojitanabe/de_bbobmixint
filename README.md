# de_bbobmixint
This repository provides the Python code to reproduce experimental results shown in the following paper:

> Ryoji Tanabe, **Benchmarking Parameter Control Methods in Differential Evolution for Mixed-Integer Black-Box Optimization**, GECCO2024, pdf

This repository is based on the COCO framework implemented in Python (https://github.com/numbbo/coco):

> Nikolaus Hansen, Anne Auger, Raymond Ros, Olaf Mersmann, Tea Tusar, and Dimo Brockhoff. 2021. COCO: a platform for comparing continuous optimizers in a black-box setting. Optim. Methods Softw. 36, 1 (2021), 114-144. https://doi.org/10.1080/10556788.2020.1808977

# Requirements

This code at least require Python 3 and numpy. To perform a benchmarking of DE on the BBOB test functions, this code require the module "cocoex" provided by COCO. The module "cocopp" is also necessary for postprocessing experimental data. For details, please see https://github.com/numbbo/coco. 

# Usage

## Simple example

The following command runs a DE with the parameter control method in jDE and the rand/2 mutation strategy on the 10-dimensional Sphere function:

```
$ python de.py
```

The following command runs a DE with the parameter control method in jDE, the rand/2 mutation strategy, and the Baldwin repair method on all the 5-, 10-, 20-, 40-, 80-, 160-dimensional 24 mixed-integer BBOB test functions. The maximum number of function evaluations is 10000 * dimensionality. Results are recorded into exdata/p-j\_rand\_2\_Baldwin. 

```
$ python de_bbob.py --de_alg jde --de_strategy rand_2 --int_handling Baldwin --output_folder p-j_rand_2_Baldwin
```
