# Code for pKSD
## How to install?
Packages that this programme depends on are listed in `setup.py`. Before running any scripts, run the following to install the current package and the dependencies
```bash
pip install -e .
```
This programme **also** depends on the [kgof]() package for the implementation of the FSSD test ([Jitkrittum et al., 2017. An interpretable linear-time kernel goodness-of-fit test](http://papers.neurips.cc/paper/6630-a-linear-time-kernel-goodness-of-fit-test.pdf)), which can be installed by running
```bash
pip install git+https://github.com/wittawatj/kernel-gof.git
```

## Examples
See `example_gaussian_mix.ipynb` for an example of how to use the spKSD and ospKSD tests for a given target distribution and a given sample. *All dependencies must be installed first* following the instruction in **"How to install?"**.

To reproduce results in the paper, run e.g.,
```bash
# mixture of two gaussians example
sh sh_scripts/run_bimodal.sh
```
Results will be stored in `res/bimodal`. Other experiments can be reproduced similarly by changing `run_bimodal.sh` to `run_rbm.sh`, `run_t-banana.sh` and `run_sensors.sh`.

## Folder structure

```bash
.
├── src                           # Source files for pKSD and benchmarks
├── sh_scripts                    # Shell scripts to run experiments
├── res                           # Folder to store results
├── figs                          # Folder to store figures
├── experiments.py                # Main script for generating results
├── example_gaussian_mix.ipynb    # Demonstration for how to use pKSD tests (spKSD and ospKSD)
├── setup.py                      # Setup file for easy-install of pKSD
└── README.md
```

