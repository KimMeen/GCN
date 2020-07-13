# Graph Convolutional Networks (GCN)

A PyTorch implementation of the paper https://arxiv.org/abs/1609.02907

### Codes

The folder contains a GCN implementation based on DGL.

Three training scripts are available to run on Cora, Citeseer, and Pubmed.

### Results

Run with following:

~~~bash
```
python train_cora.py
```
~~~

```
* cora: ~0.823 (paper: 0.815)
* citeseer: ~0.705 (paper: 0.703)
* pubmed: ~0.800 (paper: 0.790)
```

 

