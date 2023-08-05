# BiMGCL
![](https://zenodo.org/badge/doi/10.5281/zenodo.7932312.svg)

Source code for paper **BiMGCL: Rumor Detection via Bi-directional Multi-level Graph Contrastive Learning**, PeerJ Computer Science, 2023.

## Dependencies
python 3.7 
pytorch 1.8.1
pytorch_geometric 1.7.0


## Usage
create "Twitter15graph" folder and "Twitter16graph" folder in the data folder
```
python ./Process/getTwittergraph.py Twitter15 # pre-process the Twitter15 dataset
python ./Process/getTwittergraph.py Twitter16 # pre-process the Twitter16 dataset

python ./Model/train.py Twitter15 100 # Run BiMGCL for 100 epochs on Twitter15 dataset
python ./Model/train.py Twitter16 100 # Run BiMGCL for 100 epochs on Twitter16 dataset
```

## Dataset
The datasets used in the experiments were based on the two publicly available Twitter datasets released by Ma et al. (2017):

    Jing Ma, Wei Gao, Kam-Fai Wong. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL 2017.

In the 'data' folder we provide the pre-processed data files used for our experiments. The raw datasets can be downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0. To learn more about the dataset, please refer to [RvNN](https://github.com/majingCUHK/Rumor_RvNN) for more details.

**If you find this code useful, please cite our paper.**
