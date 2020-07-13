# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:29:18 2020

@author: Ming Jin

Simplified GCN training on Cora based on DGL official examples:
https://github.com/dmlc/dgl/tree/0.4.x/examples/pytorch/gcn
"""

import torch
import torch.nn as nn
import networkx as nx
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

import time
import numpy as np

from GCN import GCN

'''
define a 2-layer GCN network
'''
class GCNs(nn.Module):
    
    def __init__(self):
        super(GCNs, self).__init__()
        self.gcn1 = GCN(1433, 16, nn.ReLU())  # 1433 features
        self.gcn2 = GCN(16, 7, nn.LogSoftmax(dim=1))  # 7 classes
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        h1 = self.gcn1(g, features)
        self.dropout(h1)
        h2 = self.gcn2(g, h1)
        return h2

# initial the network
GCNs = GCNs()
    
'''
Cora dataset function
'''
def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop, A^hat = A + I in the paper 
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    # return graph, node features, labels, and training mask
    return g, features, labels, mask, val_mask, test_mask


'''
train a 2-layer GCN on Cora dataset
'''

g, features, labels, mask, val_mask, test_mask = load_cora_data()

optimizer = torch.optim.Adam(GCNs.parameters(), lr=1e-2, weight_decay=5e-4)
loss_fcn = torch.nn.NLLLoss()  # Notice that nn.CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)  # predicted class index
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

dur = []

for epoch in range(200):
    GCNs.train()
    if epoch >=3:
        t0 = time.time()
        
    optimizer.zero_grad()
    logits = GCNs(g, features)
    loss = loss_fcn(logits[mask], labels[mask])

    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)
    
    if epoch % 1 == 0:
        acc = evaluate(GCNs, g, features, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

test_acc = evaluate(GCNs, g, features, labels, test_mask)
print("\nTest accuracy {:.2%}".format(test_acc))