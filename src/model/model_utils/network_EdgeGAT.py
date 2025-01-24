import torch.nn as nn
import torch
from typing import Optional
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

def MLP(channels: list, do_bn=True, on_last=False):
   """ Multi-layer perceptron """
   n = len(channels)
   layers = []
   offset = 0 if on_last else 1
   for i in range(1, n):
       layers.append(
           torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
       if i < (n - offset):
           if do_bn:
               layers.append(torch.nn.BatchNorm1d(channels[i]))
           layers.append(torch.nn.ReLU())
   return torch.nn.Sequential(*layers)


def build_mlp(dim_list,use_bias=True, activation='relu', do_bn=False,
             dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
       dim_in, dim_out = dim_list[i], dim_list[i + 1]
       layers.append(torch.nn.Linear(dim_in, dim_out,bias=use_bias))
       final_layer = (i == len(dim_list) - 2)
       if not final_layer or on_last:
           if do_bn:
               layers.append(torch.nn.BatchNorm1d(dim_out))
           if activation == 'relu':
               layers.append(torch.nn.ReLU())
           elif activation == 'leakyrelu':
               layers.append(torch.nn.LeakyReLU())
       if dropout > 0:
           layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)

class EdgeGAT(MessagePassing):
   def __init__(self, dim_node,dim_edge,dim_atten,num_heads,num_layers , atten='hadamard_product',out_dim_node=256,
                out_dim_edge=256,use_bias=True,aggr='sum',flow='target_to_source',mode='train',layer_norm=False, batch_norm=True, residual=True):
       super().__init__()
       self.aggr = aggr
       self.flow = flow
       self.num_heads = num_heads
       self.dim_node = dim_node
       self.dim_edge = dim_edge
       self.dim_attn = dim_atten
       self.atten = atten
       self.out_dim_node = out_dim_node
       self.out_dim_edge = out_dim_edge

       self.d_n = dim_node // num_heads
       self.d_e = dim_atten // num_heads
       self.d_v = dim_atten // num_heads

       self.training = mode =='train'
       self.dropout = 0.1

       self.residual = residual
       self.layer_norm = layer_norm
       self.batch_norm = batch_norm
       if use_bias:
           self.proj_q = build_mlp([dim_node,dim_node],use_bias=True)
           self.proj_k = build_mlp([dim_node,dim_node],use_bias=True)
           self.proj_v = build_mlp([dim_node,dim_atten],use_bias=True)
           self.proj_e = build_mlp([dim_edge,dim_atten],use_bias=True)
       else:
           self.proj_q = build_mlp([dim_node, dim_node],use_bias=False)
           self.proj_k = build_mlp([dim_node, dim_node],use_bias=False)
           self.proj_v = build_mlp([dim_node, dim_atten],use_bias=False)
           self.proj_e = build_mlp([dim_edge, dim_atten],use_bias=False)

       if atten == 'concat':
           self.proj_N2N = MLP([self.d_n*2, dim_atten//self.num_heads])#build_mlp([dim_node * 2, dim_node, dim_atten])
       elif atten == 'hadamard_product':
           self.proj_N2N = MLP([self.d_n,dim_atten//self.num_heads])#build_mlp([dim_node, dim_node, dim_atten])

       # out_node_dim = dim_node
       # out_edge_dim = dim_edge
       self.O_h = nn.Linear(dim_atten, dim_atten)
       self.O_e = nn.Linear(dim_atten, dim_atten)

       if self.layer_norm:
           self.layer_norm1_h = nn.LayerNorm(dim_atten)
           self.layer_norm1_e = nn.LayerNorm(dim_atten)

       if self.batch_norm:
           self.batch_norm1_h = nn.BatchNorm1d(dim_atten)
           self.batch_norm1_e = nn.BatchNorm1d(dim_atten)

       # FFN for h
       self.FFN_h_layer1 = nn.Linear(dim_atten, dim_atten * 2)
       self.FFN_h_layer2 = nn.Linear(dim_atten * 2, out_dim_node)

       # FFN for e
       self.FFN_e_layer1 = nn.Linear(dim_atten, dim_atten * 2)
       self.FFN_e_layer2 = nn.Linear(dim_atten * 2, out_dim_edge)

       if self.layer_norm:
           self.layer_norm2_h = nn.LayerNorm(dim_atten)
           self.layer_norm2_e = nn.LayerNorm(dim_atten)

       if self.batch_norm:
           self.batch_norm2_h = nn.BatchNorm1d(dim_atten)
           self.batch_norm2_e = nn.BatchNorm1d(dim_atten)

   def forward(self, x, edge_feature, edge_index):
       '''
       :param x: (nodeNum, dim_node)
       :param edge_feature: (edgeNum, dim_edge)
       :param edge_index: ( 2, edgeNum)
       :return: updated (node & edge) features
       '''
       h_in1 = x  # for first residual connection
       e_in1 = edge_feature  # for first residual connection

       E_h , agg_edge_indicator = self.twin_EdgeAttention_for_nodes(edge_feature,edge_index,size=x.size(0))
       node_feats, edge_feats =  self.propagate(edge_index, x=x, edge_feature=E_h,agg_edge_indicator = agg_edge_indicator)

       h = F.dropout(node_feats, self.dropout, training=self.training)
       e = F.dropout(edge_feats, self.dropout, training=self.training)

       h = self.O_h(h)
       e = self.O_e(e)

       if self.residual:
           h = h_in1 + h  # residual connection
           e = e_in1 + e  # residual connection

       if self.layer_norm:
           h = self.layer_norm1_h(h)
           e = self.layer_norm1_e(e)

       if self.batch_norm:
           h = self.batch_norm1_h(h)
           e = self.batch_norm1_e(e)

       h_in2 = h  # for second residual connection
       e_in2 = e  # for second residual connection

       # FFN for h
       h = self.FFN_h_layer1(h)
       h = F.relu(h)
       h = F.dropout(h, self.dropout, training=self.training)
       h = self.FFN_h_layer2(h)

       # FFN for e
       e = self.FFN_e_layer1(e)
       e = F.relu(e)
       e = F.dropout(e, self.dropout, training=self.training)
       e = self.FFN_e_layer2(e)

       return h, e
   def twin_EdgeAttention_for_nodes(self,edge_feature,edge_index,size):
       proj_e = self.proj_e(edge_feature) #(num_edges,dim_atten)
       raw_out_row = scatter(proj_e, edge_index[0, :], dim=0, reduce='mean',dim_size=size)  # (num_nodes,dim_atten)
       raw_out_col = scatter(proj_e, edge_index[1, :], dim=0, reduce='mean',dim_size=size)  # (num_nodes, dim_atten)

       raw_out_row = raw_out_row.view(-1, self.d_e, self.num_heads) #(num_nodes, dim_atten//num_heads, num_heads)
       raw_out_col = raw_out_col.view(-1, self.d_e, self.num_heads) #(num_nodes, dim_atten//num_heads, num_heads)

       agg_edge_indicator_logits = raw_out_row * raw_out_col #(num_nodes, dim_atten//num_heads, num_heads)
       agg_edge_indicator = torch.sigmoid(agg_edge_indicator_logits) #(num_nodes, dim_atten//num_heads, num_heads)
       return proj_e , agg_edge_indicator
   def message(self, x_i:Tensor, x_j:Tensor,edge_feature : Tensor, agg_edge_indicator :Tensor) ->Tensor:
       #x_i : subject
       #x_j : object
       Q_h = self.proj_q(x_i).view(-1,self.d_n,self.num_heads) #(num_edges, dim_node//num_heads, num_heads)
       K_h = self.proj_k(x_j).view(-1,self.d_n,self.num_heads) #(num_edges, dim_node//num_heads, num_heads)
       V_h = self.proj_v(x_j) #(num_edges, dim_node_atten)

       if self.atten == 'concat':
           # concat --> dot-product attention
           node_correlation_matrix = self.proj_N2N(torch.cat([Q_h,K_h],dim=1)) #(num_edges, dim_node_atten// num_heads, num_heads)
           node_corrlation_weight = node_correlation_matrix.softmax(1) #(num_edges, dim_node_atten// num_heads, num_heads)
       elif self.atten == 'hadamard_product':
           #hadamard_product --> vector attention
           node_correlation_matrix = self.proj_N2N(Q_h * K_h) #(num_edges, dim_node_atten// num_heads, num_heads)
           node_correlation_weight = torch.sigmoid(node_correlation_matrix) #(num_edges, dim_node_atten// num_heads, num_heads)

       node_feat = torch.einsum('bm,bm->bm', node_correlation_weight.reshape_as(V_h), V_h)# (num_edges, dim_node_atten)
       return node_feat, agg_edge_indicator, node_correlation_weight,edge_feature

   def aggregate(self, x: Tensor, index: Tensor,
                 ptr: Optional[Tensor] = None,
                 dim_size: Optional[int] = None) -> Tensor:
       # x = [node_feat, agg_edge_indicator, edge_feature, node_correlation_weight]
       agg_node_feat = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr) #(num_nodes, dim_atten//num_heads, num_heads)
       update_node_feature = torch.einsum('bm,bm->bm', x[1].reshape_as(agg_node_feat),
                    agg_node_feat)
       update_edge_feat = torch.einsum('bm,bm->bm', x[2].reshape_as(x[-1]), x[-1])  # (num_edges, dim_atten)

       return (update_node_feature,update_edge_feat)


   def update(self,update_feat):
       #update_feat[0] : node_feature
       # update_feat[1] : edge_feature
       return update_feat[0], update_feat[1]



if __name__ == '__main__':
   node_feature = torch.randn((4,18))
   edge_feature = torch.randn((8,18))
   edge_index = torch.randint(0,4,size=(2,8))
   model = EdgeGAT(dim_node = 18, dim_edge=18,dim_atten=18,num_heads=1,num_layers=2,atten='hadamard_product',out_dim_node = 512, out_dim_edge = 512)

   result = model(node_feature,edge_feature,edge_index)
   print(result)


