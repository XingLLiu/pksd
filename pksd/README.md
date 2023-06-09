# Source code
```bash
.
├── kgof                    # Code for KSDAgg and RBM model, adapted from ./kgof of https://github.com/wittawatj/kernel-gof 
├── bootstrap.py            # Implementation of the bootstrap procedure 
├── find_modes.py           # Implementation of the estimation procedure for mode locations and Hessians using BFGS
├── kernel.py               # Positive definite kernels
├── ksd.py                  # KSD and pKSD
├── langevin.py             # Markov transition kernels
├── models_np.py            # Examples in numpy
├── models.py               # Examples in tensorflow
├── sensors_locations.R     # Rscript for reproducing the sensors location example of Tak et al. 2016
├── sensors.py      # Run pKSD and benchmarks on sensors location example
└── README.md
```