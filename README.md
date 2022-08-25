# Simple Message Passing for Link Prediction (SMPLP)

Support five message passing and two with JKnet:
SAGE, LRGA, GCN, SGC, TAGC, SGC+JKnet, SAGE+JKnet

**[SAGE](https://arxiv.org/abs/1706.02216)**
**[LRGA](https://arxiv.org/abs/2006.07846)**
**[GCN](https://arxiv.org/abs/1609.02907)**
**[SGC](https://arxiv.org/pdf/1902.07153.pdf)**
**[TAGC](https://arxiv.org/abs/1710.10370)**
**[JKnet](https://arxiv.org/abs/1806.03536)**


Support five link prediction methods: MLP/Dot/Bilinear Dot/MLP Cat/MLP Bilinear


## Environment:

- Dependencies: 
```{bash}
python==3.8
torch==1.10.1+cu102
torch-geometric==2.0.4
ogb==1.3.4
```
- GPU: Tesla V100 (32GB)

## OGB Dateset:
The dataset [ogbl-vessel](https://ogb.stanford.edu/docs/linkprop/#ogbl-vessel) can be download and placed in `./dataset/ogbl_vessel/`.

The dataset [ogbl-collab](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab) can be download and placed in `./dataset/ogbl-collab/`.


## Results on OGB Challenges
Performance on **ogbl-vessel** (10 runs):

| Methods   | Test Acc  | Valid Acc  |
|  :----  | ---- | ---- |
| TAGConv (1-layers) |  OOM | OOM  |
| LRGA (1-layers) |  OOM | OOM  |
| SGC (3-layers) |  54.31 ± 23.79 | 54.33 ± 23.89  |
| SGC (3-layers w/o normalize) |  50.09 ± 0.11 | 50.10 ± 0.11  |


### ogbl-collab

Performance on **ogbl-collab** (10 runs):

| Methods   |  Test Hits@50  | Valid Hits@50  |
|  :----  | ---- | ---- |
| LRGA (1-layers) |  0.6909 ± 0.0055  | 1.0000 ± 0.0000  |


## Reference
[1] https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/vessel
[2] https://github.com/zhitao-wang/PLNLP
[3] https://github.com/omri1348/LRGA

