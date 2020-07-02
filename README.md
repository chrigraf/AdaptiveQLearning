# Adaptive Discretization for Reinforcement Learning
This repository contains a reference implementation for the algorithms
appearing in the papers \[2\] for model-free Q learning in continuous spaces and \[3\] for model-based value iteration in continuous spaces.

### Dependencies
The code has been tested in `Python 3.7.7` and depends on a number of Python
packages. For the core implementation, found under `src/` we include the following files:

* `environment.py`: defines an environment the agent interacts in
* `agent.py`: defines the agent
* `experiment.py`: defines an experiment and saves data

These implementations are adapted from TabulaRL developed by Ian Osband \[1\] extended to continuous state action spaces.  They serve as a test-bed for testing an RL algorithm where new algorithms are tested by implementing several key functions in the `agent` class.

For the remaining scripts which are aimed to reproduce some of the experimental
results found in the paper and can be found in the root directory of this repo,
the following packages are required:

* numpy 1.18.1
* matplotlib 3.1.3
* pandas 1.00.3
* seaborn 0.10.1

### Quick Tour

We offer implementations for four algorithms.  First, an adaptive discretization for model-free Q learning from \[2\] and its corresponding model-free epsilon net algorithm from \[4\].  We also include implementation of AdaMB from \[3\] and an epsilon net UCBVI algorithm.  All algorithms are implemented with a state space and action space of [0,1] in mind, but for an extension to higher-dimensional space please see the `multi_dimension` subfolder.

The following files implement the different algorithms:
* `adaptive_Agent.py`: implements the Adaptive Discretization algorithm
* `eNet_Agent.py`: implements the discrete algorithm on the epsilon net
* `data_Agent.py`: implements the heuristic algorithms discussed for the ambulance problem

These agents are imported and used in the different tests.  To run the experiments used in the papers the following two files can be used.
* `run_oil_experiments_save_data.py`
* `run_ambulance_experiments_save_data.py`

Each file has parameters at the top which can be changed in order to replicate the parameters considered for each experiment in the paper.  We also include a how-to jupyter notebook walking through the code and an example in `walkthrough.ipynb`.

### Bibliography

\[1\]: Ian Osband, TabulaRL (2017), Github Repository. https://github.com/iosband/TabulaRL

\[2\]: Sean R. Sinclair, Siddhartha Banerjee, Christina Lee Yu. *Adaptive Discretization for Episodic Reinforcement Learning in Metric Spaces.* Available
[here](https://arxiv.org/abs/1910.08151).

\[3\]: Sean R. Sinclair, Tianyu Wang, Gauri Jain, Siddhartha Banerjee, Christina Lee Yu. *Adaptive Discretization for Model Based Reinforcement Learning.* Available
[Coming Soon]().

\[4\]: Zhao Song, Wen Sun. *Efficient Model-free Reinforcement Learning in Metric Spaces.* Available [here](https://arxiv.org/abs/1905.00475).
