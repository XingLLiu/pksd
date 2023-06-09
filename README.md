# Code for pKSD
## How to install?
Before running any scripts, run the following to install the current package and the dependencies. Packages that this programme depends on are listed in `setup.py`. 
```bash
pip install git+https://github.com/XingLLiu/pksd.git
```

This programme **also** depends on the [kgof]() package for the implementation of the FSSD test ([Jitkrittum et al., 2017. An interpretable linear-time kernel goodness-of-fit test](http://papers.neurips.cc/paper/6630-a-linear-time-kernel-goodness-of-fit-test.pdf)), which can be installed by running
```bash
pip install git+https://github.com/wittawatj/kernel-gof.git
```
Code for KSDAgg ([Schrab et al., 2022. KSD Aggregated Goodness-of-fit Test](https://arxiv.org/abs/2202.00824)) is copied from this [Github repo](https://github.com/antoninschrab/ksdagg) and held in `pksd/kgof`.

## Examples
This package can then be loaded as a python module with
```python
import pksd
```
See `example_gaussian_mix.ipynb` for an example of how to use the spKSD and ospKSD tests for a given target distribution and a given sample. *All dependencies must be installed beforehand* following the instruction in **"How to install?"**.

To reproduce the results in the paper, run e.g.,
```bash
# mixture of two gaussians example
sh sh_scripts/run_bimodal.sh
```
Results will be stored in `res/bimodal`. Other experiments can be reproduced similarly by changing `run_bimodal.sh` to `run_rbm.sh`, `run_t-banana.sh` or `run_sensors.sh`.

## Folder structure

```bash
.
├── pksd                          # Source files for pKSD and benchmarks
├── sh_scripts                    # Shell scripts to run experiments
├── res                           # Folder to store results
├── figs                          # Folder to store figures
├── experiments.py                # Main script for generating results
├── experiment_sensor.py          # Main script for running the sensor localisation example
├── example_gaussian_mix.ipynb    # Demonstration for how to use the pKSD tests (spKSD and ospKSD) and code for producing the thumbnail
├── setup.py                      # Setup file for easy-install of pKSD
└── README.md
```

