# AlphaTensor-Quantum

This repository accompanies the paper

> Ruiz, F. J. R. et al. Quantum Circuit Optimization with AlphaTensor.
*Nature Machine Intelligence* (2024).

There are two directories:

- `/src` contains the code developed for this project pertaining to the
reinforcement learning algorithm, along with a subdirectory `/src/demo`
containing a runnable demo that connects this code with a simplified version of
[AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4), built using
the [MCTX](https://github.com/google-deepmind/mctx) package.

- `/decompositions` contains the outputs of AlphaTensor-Quantum reported in the
paper in the form of tensor decompositions, along with a Colab showing how to
load and inspect them. Decompositions are stored in `.npz` files containing a
Python dictionary mapping from the circuit name (a string) to a boolean Numpy
array of shape `(num_decompositions, num_factors, tensor_size)`, where we report
*at most* `num_decompositions = 10` non-equivalent decompositions (equivalence
is determined solely based on factor permutations).


## Installation

- `/src`: No installation required. This folder does not contain runnable code.

- `/src/demo`: A machine with Python 3 installed is required, ideally with a
hardware accelerator such as a GPU or TPU. The required dependencies (assuming
an Nvidia GPU is available) can be installed by executing
`pip3 install -r alphatensor_quantum/src/demo/requirements.txt` (this has been
tested with Python 3.11.9 on an Nvidia Quadro P1000 GPU).

- `/decompositions`: No installation required. The provided notebook can be
opened and run in Google Colab.


## Usage

- `/src`: This folder does not contain runnable code.

- `/src/demo`: Execute `python3 -m alphatensor_quantum.src.demo.run_demo` on
the command line, from the parent directory that contains the
`alphatensor_quantum` repository as a subdirectory.

- `/decompositions`: The notebook `load_decompositions.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/alphatensor_quantum/blob/master/decompositions/load_decompositions.ipynb).
When running the notebook, you will be asked to upload certain files containing
the decompositions; please select the appropriate `.npz` file from the folder.


## Citing this work

If you use the code or data in this package, please cite:

```latex
@article{alphatensor_quantum,
      author={Ruiz, Francisco J. R. and Laakkonen, Tuomas and Bausch, Johannes and Balog, Matej and Barekatain, Mohammadamin and Heras, Francisco J. H. and Novikov, Alexander and Fitzpatrick, Nathan and Romera-Paredes, Bernardino and van de Wetering, John and Fawzi, Alhussein and Meichanetzidis, Konstantinos and Kohli, Pushmeet},
      title={Quantum Circuit Optimization with {A}lpha{T}ensor},
      journal = {Nature Machine Intelligence (Under Review)},
      year={2024},
}
```


## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
