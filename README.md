# BiMGCL
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
We use Twitter15 and Twitter16 dataset for the experiment.    
To learn more about the dataset, please refer to [RvNN](https://github.com/majingCUHK/Rumor_RvNN) for more details.

**If you find this code useful, please cite our paper.**



