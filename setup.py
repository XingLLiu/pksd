# run "pip install -e ." to setup
from setuptools import setup, find_packages

setup(
    name="pksd",
    
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.1.0",

    # Choose your license
    license="MIT",

    # What does your project relate to?
    keywords="hypothesis-test kernel-methods machine-learning AI goodness-of-fit",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(include=["pksd", "pksd.*"]),

    # See https://www.python.org/dev/peps/pep-0440/#version-specifiers
    python_requires=">= 3.8",

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy",
        "tensorflow",
        "tensorflow_probability",
        "pandas",
        "matplotlib",
        "jupyter",
        "sklearn",
        "tqdm",
        "seaborn",
        "future",
        "autograd"
    ],


    # install kgof of Jitkirittum et al., 2017 An interpretable linear-time
    # kernel goodness-of-fit test 
    dependency_links=["https://github.com/wittawatj/kernel-gof/tarball/master#egg=kgof-0.1.0"]
)
