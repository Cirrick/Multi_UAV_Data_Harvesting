# Model-aided FedQMIX
## Introduction
This repository contains the implementation of the paper [Model-aided Federated Reinforcement Learning for Multi-UAV Trajectory Planning in IoT Networks](https://arxiv.org/abs/2306.02029).
## Requirements

- python 3.7.15 or newer
- torch 1.12.1 or newer
- matplotlib 3.5.3 or newer
- pyswarms 1.3.0 or newer

## Acknowledgement
The implementation of the QMIX and IQL algorithms was heavily based on [starry-sky6688/marl-algorithms](https://github.com/starry-sky6688/marl-algorithms). 

The simulation environment was implemented with inspirations derived from the [SMAC](https://github.com/oxwhirl/smac).

## Quick Start
### Train
```shell
$ python main.py --map --federated --model --alg --num_agents --tag --device

--map: the map name, RBM/RDM
--federated: whether to use federated learning
--model: whether to use model-aided learning
--alg: the algorithm name, qmix/iql
--num_agents: the number of agents, which is equal to the number of workers in federated learning
--tag: the tag of the experiment
--device: the device to run the experiment, cpu/cuda
```
For example, if you want to train a model-aided FedQMIX on the RDM map with 3 agents, you can run the following command:
```shell    
$ python main.py --map=RDM --federated=True --model=True --alg=qmix --num_agents=3 --tag=model_aided_fedqmix
```
### Evaluate
If you want to evaluate the above trained model, you can run the following command:
```shell
$ python evaluate.py --map=RDM --model=True --alg=qmix --tag=model_aided_fedqmix --num_agents=3 
```
### Plot
Once the model training is complete, you can visualize the results using the `Plot.py` script.

### Hints
We also provide an algorithm to improve the model learning performance using kmeans, you can try it by setting `--model=True --sample_method=kmeans`.

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