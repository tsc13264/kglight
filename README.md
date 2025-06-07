# Introduction
KGLight is a reinforcement learning-based traffic signal control framework that leverages an Urban Knowledge Graph (UKG) to incorporate rich urban semantic information for more efficient and scalable large-scale signal management.

Our repository is based on the [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) traffic signal control framework and integrates compatibility with the [CBLab](https://github.com/caradryanl/CityBrainLab) traffic simulator into this framework.

# Quick Start

## Requirement
<br />

Make sure to install PyTorch and related libraries first, then install other dependencies listed in requirements.txt.

```
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Run Model

You can start the program with the following command, and various arguments are supported for custom configuration.

```
python run.py
```