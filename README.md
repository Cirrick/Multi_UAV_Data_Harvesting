# Model-aided FedQMIX
## Introduction
This repository contains the implementation of the paper [Model-aided Federated Reinforcement Learning for Multi-UAV Trajectory Planning in IoT Networks](https://arxiv.org/abs/2306.02029).
## Requirements

- python 3.7.15 or newer
- numpy 1.21.5 or newer
- torch 1.12.1 or newer
- matplotlib 3.5.3 or newer
- pyswarms 1.3.0 or newer

## Acknowledgement

We gratefully acknowledge the work [Model-aided Deep Reinforcement Learning for Sample-efficient UAV Trajectory Design in IoT Networks](https://ieeexplore.ieee.org/abstract/document/9685774), which we have expanded upon to accommodate multi-UAV scenarios. Our implementation of the QMIX and IQL algorithms was substantially based on [starry-sky6688/marl-algorithms](https://github.com/starry-sky6688/marl-algorithms). Additionally, our simulation environment design drew inspiration from the [SMAC](https://github.com/oxwhirl/smac).

## Quick Start
### Train
```shell
$ python main.py --map --federated --model --alg --n_agents --tag --total_episodes --device

--map: the map name, RBM/RDM
--federated: whether to use federated learning
--model: whether to use model-aided learning
--alg: the algorithm name, qmix/iql
--n_agents: the number of agents, which is equal to the number of workers in federated learning
--tag: the tag of the experiment
--total_episodes: the total number of collected episodes
--device: the device to run the experiment, cpu/cuda
```
For example, if you want to train a model-aided FedQMIX on the RDM map with 3 agents, you can run the following command:
```shell    
$ python main.py --map=RDM --federated=True --model=True --alg=qmix --n_agents=3 --tag=model_aided_fedqmix --total_episodes=30000
```
### Evaluate
If you want to evaluate the above trained model, you can run the following command:
```shell
$ python evaluate.py --map=RDM --model=True --alg=qmix --tag=model_aided_fedqmix --n_agents=3 
```
### Plot
Once the model training is complete, you can visualize the results using the `Plot.py` script.

### Hints
We also provide an algorithm to improve the model learning performance by using k-means algorithm, you can try it by setting `--model=True --sample_method=kmeans`.

## Reference
If you find this repository useful, please cite our paper:
```
@article{chen2023model,
  title={Model-aided Federated Reinforcement Learning for Multi-UAV Trajectory Planning in IoT Networks},
  author={Chen, Jichao and Esrafilian, Omid and Bayerlein, Harald and Gesbert, David and Caccamo, Marco},
  journal={arXiv preprint arXiv:2306.02029},
  year={2023}
}
```
