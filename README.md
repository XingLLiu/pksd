# Code for pKSD
[Paper](https://arxiv.org/abs/2304.14762) | [Poster](https://github.com/XingLLiu/pksd/blob/main/_asset/poster_summer_school.pdf)

## How to install?
### Install as a package
Before running any scripts, run the following to install the current package and the dependencies. Packages that this programme depends on are listed in `setup.py`. 
1. Run the follwing to install the this repo as a package called `pksd`.
```bash
pip install git+https://github.com/XingLLiu/pksd.git
```
2. Run the following to install the [kgof](https://github.com/wittawatj/kernel-gof/tree/master) package for the implementation of the FSSD test ([Jitkrittum et al., 2017. An interpretable linear-time kernel goodness-of-fit test](http://papers.neurips.cc/paper/6630-a-linear-time-kernel-goodness-of-fit-test.pdf)), which is required by our package.
```bash
pip install git+https://github.com/wittawatj/kernel-gof.git
```
Code for KSDAgg ([Schrab et al., 2022. KSD Aggregated Goodness-of-fit Test](https://arxiv.org/abs/2202.00824)) is copied from this [Github repo](https://github.com/antoninschrab/ksdagg) and held in `pksd/kgof`, and hence it does not require installation. This is to customise the output of their test so that results can be copmared with the other tests.

### Install dependencies only
Alternatively, to install only the dependencies but not as a package, run
```bash
pip install -r requirements.txt
```

## Examples
This package can be loaded as a python module with
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
