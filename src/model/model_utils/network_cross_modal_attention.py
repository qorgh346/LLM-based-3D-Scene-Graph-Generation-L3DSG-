
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
   '''
   Scaled dot-product attention
   '''

   def __init__(self, d_model, d_k, d_v, h):
       '''
       :param d_model: Output dimensionality of the model
       :param d_k: Dimensionality of queries and keys
       :param d_v: Dimensionality of values
       :param h: Number of heads
       '''
       super(ScaledDotProductAttention, self).__init__()
       self.fc_q = nn.Linear(d_model, h * d_k)
       self.fc_k = nn.Linear(d_model, h * d_k)
       self.fc_v = nn.Linear(d_model, h * d_v)
       self.fc_o = nn.Linear(h * d_v, d_model)

       self.d_model = d_model
       self.d_k = d_k
       self.d_v = d_v
       self.h = h

       self.init_weights()

   def init_weights(self):
       nn.init.xavier_uniform_(self.fc_q.weight)
       nn.init.xavier_uniform_(self.fc_k.weight)
       nn.init.xavier_uniform_(self.fc_v.weight)
       nn.init.xavier_uniform_(self.fc_o.weight)
       nn.init.constant_(self.fc_q.bias, 0)
       nn.init.constant_(self.fc_k.bias, 0)
       nn.init.constant_(self.fc_v.bias, 0)
       nn.init.constant_(self.fc_o.bias, 0)

   def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul', use_knn=False):
       '''
       Computes
       :param queries: Queries (b_s, nq, d_model)
       :param keys: Keys (b_s, nk, d_model)
       :param values: Values (b_s, nk, d_model)
       :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
       :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
       :return:
       '''
       b_s, nq = queries.shape[:2]
       nk = keys.shape[1]
       #print(queries.shape, keys.shape, values.shape)
       q = self.fc_q(queries)
       #print(q.shape, b_s, nq, self.h, self.d_k)
       q = q.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
       k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
       v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

       att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
       att_map = att.clone()
       if use_knn:
           att = attention_weights
       else:
           if attention_weights is not None:
               if way == 'mul':
                   att = att * attention_weights
               elif way == 'add':
                   #print(att.shape, attention_weights.shape, '<< att shape; add')
                   att = att + attention_weights
               else:
                   raise NotImplementedError(way)
       if attention_mask is not None:
           att = att.masked_fill(attention_mask==0, -np.inf)
       att = torch.softmax(att, -1)
       out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
       out = self.fc_o(out)  # (b_s, nq, d_model)
       return out, att_map


class MultiHeadAttention(nn.Module):
   '''
   Multi-head attention layer with Dropout and Layer Normalization.
   '''

   def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                attention_module=None, attention_module_kwargs=None):
       super(MultiHeadAttention, self).__init__()
       self.identity_map_reordering = identity_map_reordering
       if attention_module is not None:
           if attention_module_kwargs is not None:
               self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
           else:
               self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m = 20)
       else:
           self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
       self.dropout = nn.Dropout(p=dropout)
       self.layer_norm = nn.LayerNorm(d_model)

       self.can_be_stateful = can_be_stateful
       if self.can_be_stateful:
           self.register_state('running_keys', torch.zeros((0, d_model)))
           self.register_state('running_values', torch.zeros((0, d_model)))

   def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul', use_knn=False, output_attn=False):
       if self.can_be_stateful and self._is_stateful:
           self.running_keys = torch.cat([self.running_keys, keys], 1)
           keys = self.running_keys

           self.running_values = torch.cat([self.running_values, values], 1)
           values = self.running_values

       if self.identity_map_reordering:
           q_norm = self.layer_norm(queries)
           k_norm = self.layer_norm(keys)
           v_norm = self.layer_norm(values)
           out, att = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, way)
           out = queries + self.dropout(torch.relu(out))
       else:
           out, att = self.attention(queries, keys, values, attention_mask, attention_weights, way, use_knn)
           out = self.dropout(out)
           out = self.layer_norm(queries + out)
       if output_attn:
           return out, att
       else:
           return out


class CrossModalAttention(nn.Module):
   def __init__(self, k_dim,g_dim,hidden_dim,output_dim,dropout=0.2):
       super(CrossModalAttention, self).__init__()
       self.query = nn.Linear(g_dim, hidden_dim)
       self.key = nn.Linear(k_dim, hidden_dim)
       self.value = nn.Linear(k_dim, hidden_dim)
       self.ffn = nn.Linear(hidden_dim, output_dim) #gcn_dim
       self.d_k = hidden_dim ** 0.5
       self.dropout = nn.Dropout(p=dropout)
       self.layer_norm = nn.LayerNorm(hidden_dim)
       self.know_layer_norm = nn.LayerNorm(k_dim)
       self.init_weights()
   def init_weights(self):
       nn.init.xavier_uniform_(self.query.weight)
       nn.init.xavier_uniform_(self.key.weight)
       nn.init.xavier_uniform_(self.value.weight)
       nn.init.xavier_uniform_(self.ffn.weight)
       nn.init.constant_(self.query.bias, 0)
       nn.init.constant_(self.key.bias, 0)
       nn.init.constant_(self.value.bias, 0)
       nn.init.constant_(self.ffn.bias, 0)

   def forward(self,pointFeat, knowFeat):
       q_norm = self.layer_norm(pointFeat)
       k_norm = self.know_layer_norm(knowFeat)
       v_norm = self.know_layer_norm(knowFeat)
       Q = self.query(q_norm)  # Query from A, shape (4, hidden_dim)
       K = self.key(k_norm)    # Key from B, shape (12, hidden_dim)
       V = self.value(v_norm)  # Value from B, shape (12, hidden_dim)

       attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k # Shape (4, 12)
       att_map = attn_scores.clone()

       attn_weights = F.softmax(attn_scores, dim=-1) # Shape (4, 12)

       attention_vector = torch.matmul(attn_weights, V) # Shape (4, hidden_dim)

       residual_feat = pointFeat + self.dropout(torch.relu(attention_vector))
       # residual_feat = torch.concat((pointFeat,attention_vector),dim=1)
       fused_feature = self.ffn(self.layer_norm(residual_feat))

       return fused_feature, att_map

if __name__ == '__main__':
   dim = 128
   pointFeat = torch.rand((4, 512))
   knowFeat = torch.rand((12, 768))

   attention_layer = CrossModalAttention(k_dim=768,g_dim=512,hidden_dim=512,output_dim=256,dropout=0.2)
   output = attention_layer(pointFeat, knowFeat)
   print(output) # Output shape will be (4, 128)


