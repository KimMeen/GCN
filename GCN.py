# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:37:02 2020

@author: Ming Jin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph

class GCN(nn.Module):
    '''
    in_feats:
        Input feature size
    out_feats:
        Output feature size
    activation:
        Applies an activation function to the updated node features
    '''
    
    def __init__(self, in_feats, out_feats, activation=None):
        super(GCN, self).__init__()
        self._in_feats = in_feats  # "C" in the paper
        self._out_feats = out_feats  # "F" in the paper
        self._activation_func = activation  # "ReLu" and "Softmax" in the paper
        
        # W^{1} and W^{2} in the paper
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))  # bias, optional
        
        # initialize the weight and bias
        self.reset_parameters()
    
        
    def reset_parameters(self):
        '''
        Reinitialize learnable parameters
        ** Glorot, X. & Bengio, Y. (2010)
        ** Critical, otherwise the loss will be NaN
        '''
        
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
            
    def forward(self, g, features):
        '''
        formular: 
            h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})
        
        Inputs:
            g: 
                The fixed graph
            features: 
                H^{l}, i.e. Node features with shape [num_nodes, features_per_node]
                
        Returns:
            rst:
                H^{l+1}, i.e. Node embeddings of the l+1 layer with the 
                shape [num_nodes, hidden_per_node]
                
        Variables:
            gcn_msg: 
                Message function of GCN, i.e. What to be aggregated 
                (e.g. Sending node embeddings)
            gcn_reduce: 
                Reduce function of GCN, i.e. How to aggregate 
                (e.g. Summing neighbor embeddings)
                
        Notice: 'h' means node feature/embedding itself, 'm' means node's mailbox
        '''
                        
        # normalize features by node's out-degrees
        out_degs = g.out_degrees().float().clamp(min=1)  # shape [num_nodes]
        norm1 = torch.pow(out_degs, -0.5)
        shape1 = norm1.shape + (1,) * (features.dim() - 1)
        norm1 = torch.reshape(norm1, shape1)
        features = features * norm1
        
        # mult W first to reduce the feature size for aggregation
        features = torch.matmul(features, self.weight)
        
        # DGLGraph.ndata: Data view of all the nodes (a.k.a node features), 
        # g.ndata['h'] is a dictionary, 'h' is the key (identifier)
        # i.e. {'h' : tensor(...)}
        g.ndata['h'] = features  # provide each node with its feature
        # define the message and reduce functions
        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.sum(msg='m', out='h')
        # message passing + update
        g.update_all(gcn_msg, gcn_reduce)
        rst = g.ndata.pop('h')
        
        # normalize features by node's in-degrees
        in_degs = g.in_degrees().float().clamp(min=1)  # shape [num_nodes]
        norm2 = torch.pow(in_degs, -0.5)
        shape2 = norm2.shape + (1,) * (features.dim() - 1)
        norm2 = torch.reshape(norm2, shape2)
        rst = rst * norm2
        
        # add bias
        rst = rst + self.bias
        
        # activation
        if self._activation_func is not None:
            rst = self._activation_func(rst)
        
        # get h^{l+1} from h^{l}
        return rst