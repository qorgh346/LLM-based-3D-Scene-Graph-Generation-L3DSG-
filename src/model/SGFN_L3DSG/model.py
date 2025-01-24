import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.network_util import MLP
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMG import MMG
from src.model.model_utils.network_EdgeGAT import EdgeGAT
from src.model.model_utils.network_PointNet import (PointNetfeat,
                                                    PointNetRelCls,
                                                    PointNetRelClsMulti,
                                                    PointNetCls)

from src.model.model_utils.network_cross_modal_attention import CrossModalAttention
from src.model.model_utils.network_PointTransformer import PointTransformerfeat

from src.utils.eva_utils_acc import (evaluate_topk_object,
                                 evaluate_topk_predicate,
                                 evaluate_triplet_topk, get_gt)
from src.utils.Losses import AsymmetricLoss
from utils import op_utils
from src.model.model_utils.network_priorKnowledge import KnowledgeModel

# BRIEF text self-attention
class TransformerEncoderLayerNoFFN(nn.Module):
    """TransformerEncoderLayer but without FFN."""

    def __init__(self, d_model, nhead, dropout):
        """Intialize same as Transformer (without FFN params)."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

class L3DSGModel(BaseModel):
    def __init__(self, config, num_obj_class, num_rel_class,obj_cls,rel_cls, dim_descriptor=11):
        '''
        3d cat location
        '''

        super().__init__('L3DSGModel', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN
        self.dim_3d_object_feature = self.mconfig.object_dim_size
        self.dim_3d_relation_feature = self.mconfig.relation_dim_size
        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.dim_point_feature = self.config.MODEL.point_feature_size
        self.momentum = 0.1
        self.model_pre = None
        self.dim_gcn = self.mconfig.gcn_feat_dim
        self.gcn_output_dim = self.mconfig.gcn_output_dim
        if self.mconfig.USE_PriorKnowledge:
            self.knowledgeModel = KnowledgeModel(config, obj_cls, rel_cls)
            self.text_projector = nn.Sequential(
                nn.Linear(768, 288),
                nn.LayerNorm(288, eps=1e-12),
                nn.Dropout(0.1)
            )
            self.lang_encoder = TransformerEncoderLayerNoFFN(d_model=288, nhead=8, dropout=0.1)

            self.text_projector, self.lang_encoder = self.pretrained_3dvg(self.text_projector, self.lang_encoder)

            if self.mconfig.Target_GRAPH == "Initial":
                self.node_atte_layer = CrossModalAttention(k_dim=288, g_dim=self.dim_3d_object_feature, hidden_dim=256, output_dim=256,
                                                           dropout=0.2)
                self.edge_atte_layer = CrossModalAttention(k_dim=288, g_dim=self.dim_3d_object_feature, hidden_dim=256, output_dim=256,
                                                           dropout=0.2)
            else:
                #Target Graph is Enhanced Graph
                self.node_atte_layer = CrossModalAttention(k_dim=288, g_dim=self.gcn_output_dim, hidden_dim=256, output_dim=256,
                                                           dropout=0.2)
                self.edge_atte_layer = CrossModalAttention(k_dim=288, g_dim=self.gcn_output_dim, hidden_dim=256, output_dim=256,
                                                           dropout=0.2)
        if mconfig.USE_PointTransformer:
            self.transformer_config = self.config.PointTransformerModel
            self.transformer_config.num_point = self.config.dataset.num_points
            self.transformer_config.input_dim = self.dim_point
            self.transformer_config.transformer_dim = 256 #256 #self.mconfig.point_feature_size
            self.transformer_config.relation_point = False
            self.obj_encoder = PointTransformerfeat(self.transformer_config)
            self.obj_encoder  = self.transformerV1_Load(self.obj_encoder,self.transformer_config.weight_path)

            self.transformer_config.input_dim = self.dim_point
            self.rel_encoder = PointTransformerfeat(self.transformer_config)
            self.rel_encoder = self.transformerV1_Load(self.rel_encoder, self.transformer_config.weight_path)
        elif mconfig.USE_PointNet:
            # Object Encoder
            self.obj_encoder = PointNetfeat(
                global_feat=True,
                batch_norm=with_bn,
                point_size=dim_point,
                input_transform=False,
                feature_transform=mconfig.feature_transform,
                out_size=self.dim_point_feature)

            self.rel_encoder = PointNetfeat(
                global_feat=True,
                batch_norm=with_bn,
                point_size=dim_point,
                input_transform=False,
                feature_transform=mconfig.feature_transform,
                out_size=self.dim_point_feature)

        self.graph_reasoning =EdgeGAT(dim_node=self.dim_3d_object_feature, dim_edge=self.dim_3d_relation_feature,
                                      dim_atten=self.dim_gcn, num_heads=4, num_layers=2,
                                      atten='hadamard_product',out_dim_node=self.gcn_output_dim,
                                      out_dim_edge=self.gcn_output_dim)

        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.mlp_3d_obj = torch.nn.Sequential(
            torch.nn.Linear(self.dim_point_feature+8, self.dim_3d_object_feature),
            torch.nn.BatchNorm1d(self.dim_3d_object_feature),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        if self.mconfig.USE_RELPOINT:
            rel_input_dim = self.dim_point_feature+11
        else:
            rel_input_dim = +11

        self.mlp_3d_rel = torch.nn.Sequential(
            torch.nn.Linear(rel_input_dim, self.dim_3d_relation_feature),
            torch.nn.BatchNorm1d(self.dim_3d_relation_feature),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        self.obj_predictor_3d = PointNetCls(k=160,in_size=256)#torch.nn.Linear(256, self.num_class)

        if mconfig.multi_rel_outputs:
            self.rel_predictor_3d = PointNetRelClsMulti(
                num_rel_class, 
                in_size=256,
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor_3d = PointNetRelCls(
                num_rel_class, 
                in_size=256,
                batch_norm=with_bn,drop_out=True)

        self.init_weight(obj_label_path=mconfig.obj_label_path, \
                         rel_label_path=mconfig.rel_label_path, \
                         adapter_path=mconfig.rel_label_path)

        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params': self.rel_encoder.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD},

            {'params': self.graph_reasoning.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD},

            {'params':self.mlp_3d_obj.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params': self.mlp_3d_rel.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD},

            {'params': self.node_atte_layer.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD},
            {'params': self.edge_atte_layer.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD},


            {'params': self.obj_predictor_3d.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD},
            {'params': self.rel_predictor_3d.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY,
             'amsgrad': self.config.AMSGRAD}
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()
        self.rel_criterion = AsymmetricLoss()
    def pretrained_3dvg(self,text_proj, text_encoder):
        weights = torch.load(self.mconfig.Lang3DVG_path)

        text_proj.load_state_dict(weights['text_projector'])
        text_encoder.load_state_dict(weights['lang_encoder'])

        for param in text_proj.parameters():
            param.requires_grad = False

        for param in text_encoder.parameters():
            param.requires_grad = False

        return text_proj, text_encoder

    def init_weight(self, obj_label_path, rel_label_path, adapter_path):
        torch.nn.init.xavier_uniform_(self.mlp_3d_obj[0].weight)
        torch.nn.init.xavier_uniform_(self.mlp_3d_rel[0].weight)

        # Hojun
        # self.clip_adapter.load_state_dict(torch.load(adapter_path, 'cpu'))

        # freeze clip adapter
        # for param in self.clip_adapter.parameters():
        #     param.requires_grad = False

    def update_model_pre(self, new_model):
        self.model_pre = new_model

    def cosine_loss(self, A, B, t=1):
        return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    
    def compute_triplet_loss(self, obj_logits_3d, rel_cls_3d, obj_logits_2d, rel_cls_2d, edge_indices):
        triplet_loss = []
        obj_logits_3d_softmax = F.softmax(obj_logits_3d, dim=-1)
        obj_logits_2d_softmax = F.softmax(obj_logits_2d, dim=-1)
        for idx, i in enumerate(edge_indices):
            obj_score_3d = obj_logits_3d_softmax[i[0]]
            obj_score_2d = obj_logits_2d_softmax[i[0]]
            sub_score_3d = obj_logits_3d_softmax[i[1]]
            sub_score_2d = obj_logits_2d_softmax[i[1]]
            rel_score_3d = rel_cls_3d[idx]
            rel_score_2d = rel_cls_2d[idx]
            node_score_3d = torch.einsum('n,m->nm', obj_score_3d, sub_score_3d)
            node_score_2d = torch.einsum('n,m->nm', obj_score_2d, sub_score_2d)
            triplet_score_3d = torch.einsum('nl,m->nlm', node_score_3d, rel_score_3d).reshape(-1)
            triplet_score_2d = torch.einsum('nl,m->nlm', node_score_2d, rel_score_2d).reshape(-1)
            triplet_loss.append(F.l1_loss(triplet_score_3d, triplet_score_2d.detach(), reduction='sum')) 
            
            
        #return torch.sum(torch.tensor(triplet_loss))
        return torch.mean(torch.tensor(triplet_loss))
    
    def forward(self,obj_scenes,rel_scenes, obj_points, rel_points, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        #[node,point_feature_dim] 256

        if istrain:
            obj_feature_3d_mimic = obj_feature[..., :self.dim_point_feature].clone()

        if self.mconfig.USE_RELPOINT:
            rel_feature = self.rel_encoder(rel_points[:,:3,:])
        # [edge,point_feature_dim] 256


        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=-1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        # rel_points
        if self.mconfig.USE_RELPOINT:
            # if use rel point
            edge_feature =  torch.cat([rel_feature, edge_feature],dim=-1)

        rel_feature = self.mlp_3d_rel(edge_feature)
        # rel_feature_3d = self.rel_encoder_3d(edge_feature)

        obj_feature = self.mlp_3d_obj(obj_feature)

        # 1st forward #
        # Graph Reasoning #
        gcn_obj_feature_3d, gcn_edge_feature_3d = self.graph_reasoning(obj_feature,rel_feature,edge_indices)

        if self.mconfig.USE_PriorKnowledge:
            ########################## 2st forward #############################
            # obj_cls_3d = self.obj_predictor_3d(gcn_obj_feature_3d)
            obj_logits_3d = self.obj_predictor_3d(gcn_obj_feature_3d)

            rel_cls_3d = self.rel_predictor_3d(gcn_edge_feature_3d)

            #Prior Knowledge Prompting & Embedding#
            with torch.no_grad():
                obj_pknowledge, obj_top1_label = self.knowledgeModel.object_knowledge_embedding(k=1,
                                                               logits=obj_logits_3d,
                                                               obj_scans= obj_scenes
                                                                )
                rel_pknowledge = self.knowledgeModel.rel_knowledge_embedding(k=1,
                                                                            edge_index = edge_indices,
                                                                            predicted_obj_label = obj_top1_label,
                                                                            rel_logits=rel_cls_3d,
                                                                             rel_scans = rel_scenes
                                                                            )

                objKnowledgeProj = self.text_projector(obj_pknowledge).unsqueeze(dim=1)
                relKnowledgeProj = self.text_projector(rel_pknowledge).unsqueeze(dim=1)

                objKnowledgeEmbed = self.lang_encoder(objKnowledgeProj).squeeze(dim=1)
                relKnowledgeEmbed = self.lang_encoder(relKnowledgeProj).squeeze(dim=1)
            ## Prior Knowledge Fusion ##

            if self.mconfig.Target_GRAPH == "Initial":

                enhanced_node_feature, node_attmap = self.node_atte_layer(obj_feature,objKnowledgeEmbed)
                enhanced_edge_feature, edge_attmap = self.edge_atte_layer(rel_feature, relKnowledgeEmbed)
                # 2nd Graph Reasoning #
                gcn_obj_feature_3d, gcn_edge_feature_3d = self.graph_reasoning(enhanced_node_feature,
                                                                               enhanced_edge_feature, edge_indices)

            else:
                gcn_obj_feature_3d, node_attmap = self.node_atte_layer(gcn_obj_feature_3d, objKnowledgeEmbed)
                gcn_edge_feature_3d, edge_attmap = self.edge_atte_layer(gcn_edge_feature_3d, relKnowledgeEmbed)




        #Final Scene Graph Prediction#
        logit_scale = self.obj_logit_scale.exp()
        obj_logits_3d = self.obj_predictor_3d(gcn_obj_feature_3d)## / final_node_feature.norm(dim=-1, keepdim=True))
        # obj_logits_3d = logit_scale * obj_logits_3d
        rel_cls_3d = self.rel_predictor_3d(gcn_edge_feature_3d)


        # x = '/home/baebro/hojun_ws/LMM based 3D Scene Graph Generation(L3DSG)/L3DSG_New/knowledge'
        # self.knowledgeModel.save_knowledge(x)


        if istrain:
            return obj_logits_3d, rel_cls_3d, obj_feature_3d_mimic, logit_scale
        else:
            return obj_logits_3d, rel_cls_3d

    def process_train(self, obj_scenes,rel_scenes, obj_points, rel_points, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    

        obj_logits_3d, rel_cls_3d, obj_feature_3d, obj_logit_scale = self(obj_scenes,rel_scenes,obj_points, rel_points, edge_indices.t().contiguous(),
                                                                          descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        loss_obj_3d = F.nll_loss(obj_logits_3d, gt_cls)
        # loss_obj_3d = torch.nn.CrossEntropyLoss(obj_logits_3d, gt_cls)
         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]

            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            # loss_rel_3d = torch.nn.BCELoss(rel_cls_3d, gt_rel_cls)
            loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls, weight=weight)
            # focal loss
            # loss_rel_3d = self.rel_criterion(rel_cls_3d, gt_rel_cls)

        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[0] = 0 # assume none is the first relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel_2d = loss_rel_3d = torch.zeros(1, device=rel_cls_3d.device, requires_grad=False)
            else:
                loss_rel_3d = F.nll_loss(rel_cls_3d, gt_rel_cls, weight = weight)
                loss_rel_2d = F.nll_loss(rel_cls_2d, gt_rel_cls, weight = weight)
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        # obj_feature_3d = obj_feature_3d / obj_feature_3d.norm(dim=-1, keepdim=True)
        #f.cosine_loss(obj_feature_3d, obj_feature_3d, t=0.8)

        # compute similarity between visual with text
        # rel_text_feat = self.get_rel_emb(gt_cls, gt_rel_cls, edge_indices)

        # edge_feature_2d = edge_feature_2d / edge_feature_2d.norm(dim=-1, keepdim=True)
        # rel_mimic_2d = F.l1_loss(edge_feature_2d, rel_text_feat)
               
        # loss = lambda_o * (loss_obj_2d + loss_obj_3d) + 3 * lambda_r * (loss_rel_2d + loss_rel_3d) + 0.1 * (loss_mimic + rel_mimic_2d)
        loss = lambda_o * loss_obj_3d + lambda_r *  loss_rel_3d
        self.backward(loss)
        
        # compute 3d metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]

        # compute 2d metric
        # top_k_obj = evaluate_topk_object(obj_logits_2d.detach(), gt_cls, topk=11)
        # top_k_rel = evaluate_topk_predicate(rel_cls_2d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        # obj_topk_2d_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        # rel_topk_2d_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel_3d.detach().item()),
                ("train/obj_loss", loss_obj_3d.detach().item()),
                ("train/logit_scale", obj_logit_scale.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2])]

            #     ("train/Obj_R1_2d", obj_topk_2d_list[0]),
            #     ("train/Obj_R5_2d", obj_topk_2d_list[1]),
            #     ("train/Obj_R10_2d", obj_topk_2d_list[2]),
            #     ("train/Pred_R1_2d", rel_topk_2d_list[0]),
            #     ("train/Pred_R3_2d", rel_topk_2d_list[1]),
            #     ("train/Pred_R5_2d", rel_topk_2d_list[2]),
            # ]
        return log
           
    def process_val(self,gt_obj_scene,gt_rel_scene, obj_points, rel_points, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
 
        obj_logits_3d, rel_cls_3d = self(gt_obj_scene,gt_rel_scene, obj_points, rel_points, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)

        # delete
        # obj_logits_2d, rel_cls_2d

        # compute metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)

        # top_k_obj_2d = evaluate_topk_object(obj_logits_2d.detach().cpu(), gt_cls, topk=11)
        # top_k_rel_2d = evaluate_topk_predicate(rel_cls_2d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_softmax=False, obj_topk=top_k_obj)
            # top_k_2d_triplet, _, _, _, _ = evaluate_triplet_topk(obj_logits_2d.detach().cpu(), rel_cls_2d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_rel, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
 
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()

    def transformerV1_Load(self,model,pretrain_path):
        model_dict = model.state_dict()
        temp_pretrained_dict = dict()
        pretrained_dict = torch.load(pretrain_path)
        for k, v in pretrained_dict['model_state_dict'].items():
            if k in model_dict and 'backbone.fc1' not in k and 'fc2' not in k.split('.')[0]:
                temp_pretrained_dict[k] = v
        model_dict.update(temp_pretrained_dict)
        model.load_state_dict(model_dict, strict=True)


            # all frozen #
            # names = name.split('.')
            # if names[1] =='transformers':
            #     if names[2] == '2':
            #         param.requires_grad = True
            # param.requires_grad = False
        # sys.exit()
        return model