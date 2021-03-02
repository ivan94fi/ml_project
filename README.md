# Machine Learning Project: re-implementation of *"Noise2Noise"* framework

## Description
This project aims to partially implement the deep learning training procedure described in "Noise2Noise: Learning Image Restoration without Clean Data" by Lehtinen *et al.* [[1]](#1).

The authors of [[1]](#1) show how it is not necessary to have high quality clean targets for succesfully training a deep learning model for the image restoration task. Actually, they report comparable (if not better) performance when using corrupted target pairs for training, eliminating the need for (often expensive) clean target acquisition.

## Installation

```shell
make conda-install

# restart the terminal or open a new shell
bash

# create the environment, with the specified pytorch/cudatoolkit versions
make create-env pytorch=XXX cudatoolkit=XXX  # see on pytorch site the possible configurations

# activate the new environment
conda activate n2n

# install the ml_project package into the environment
make install # or install-dev to make modification to source files reflected in the installed package

# run a brief check to assure that the package is installed properly and the GPU is available
make check-env
```
## Authors
* **Ivan Prosperi** - <ivan.prosperi@stud.unifi.it>

## Bibliography
<a id="1">[1]</a>: J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila. Noise2Noise: Learning image restoration without clean data. In Proc. International Conference on Machine Learning (ICML), 2018.
