
# RL vs MPC for Battery Management

## Overview
This repository explores the application of Reinforcement Learning (RL) and Model Predictive Control (MPC) for battery management systems. It contains separate implementations and comparative analyses of both techniques. The full paper can be found [here](https://arxiv.org/abs/2407.15313).


## Repository Structure
- **MPC**:
  - Scripts implementing MPC with various analysis tools like confidence intervals and horizon length studies.
  - **Directories**:
    - `data`: Data files for MPC simulations.
    - `forecaster`: Forecasting models used in MPC.
    - `mpc`: Core MPC algorithm implementations.
    - `utils`: Utilities supporting MPC operations.
- **RL**:
  - Implementation of RL agents and environments tailored for battery management.
  - `main.py`: Implementing the RL experiment.
  - **Directories**:
    - `agents`: Definitions and implementations of RL agents.
    - `data`: Data for training and evaluating RL agents.
    - `environments`: Definitions of environments for RL agents.
    - `rl_monitoring_utils`: Monitoring and evaluation tools for RL agents.

## Getting Started
To get started with either the RL or MPC modules, clone the repository, navigate to the respective directory, and review the `main_notebook.ipynb` for RL or the specific scripts in the MPC directory.

## Contributing
Contributions to either the RL or MPC sections are welcome. Please fork the repository and submit a pull request with your enhancements.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
We extend our heartfelt thanks to all contributors, data providers, and advisors who have supported this project. Your contributions have been invaluable in enhancing our understanding and implementation of battery management systems.
