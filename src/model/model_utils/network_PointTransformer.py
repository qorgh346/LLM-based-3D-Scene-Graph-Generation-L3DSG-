import torch
import torch.nn as nn

from src.model.model_utils.ptv1_utils import TransformerBlock
from src.model.model_utils.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.npoints, self.nblocks, self.nneighbor, self.d_points,self.f_relation_point = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.input_dim,cfg.relation_point

        if self.f_relation_point:
            self.npoints *= 2
            # self.d_points += 1

        self.fc1 = nn.Sequential(
            nn.Linear(self.d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.transformer_dim, self.nneighbor,self.f_relation_point)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(self.nblocks): #0,1
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(self.npoints // 4 ** (i + 1), self.nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.transformer_dim, self.nneighbor,self.f_relation_point))

    def forward(self, x):
        xyz = x[..., :self.d_points]
        # temp = self.fc1(x)
        points = self.transformer1(xyz, self.fc1(x[..., :self.d_points]))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerfeat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, d_points = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.input_dim

        self.nblocks = nblocks

    def forward(self, x):
        points, xyz_and_feat = self.backbone(x)
        point_feat = points.mean(1) #aggregation..
        # res = self.fc2(points.mean(1))
        return point_feat


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )

    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]

        return self.fc3(points)

class Transformer_cfg:
    #         npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
    def __init__(self,point_feature_size=256,input_dim=6,pretrain_path = None):
        self.num_point = 512
        self.nblocks = 2
        self.nneighbor = 16
        self.num_class = 10
        self.input_dim = input_dim
        self.transformer_dim = point_feature_size #256
        self.pretrain_path = pretrain_path



if __name__ == '__main__':
    import sys

    a = torch.load('/home/baebro/hojun_ws/3DSSG_Baek/backbone_checkpoints/best_model.pth')
    print(a['model_state_dict'])
    cfg = Transformer_cfg(pretrain_path=transformer_pretrain_path)
    model = PointTransformerfeat(cfg)

    if cfg.pretrain_path:
        pretrained_dict = torch.load(cfg.pretrain_path)
        model_dict = model.state_dict()  # 현재 신경망 상태 로드
        temp_pretrained_dict = dict()
        for k, v in pretrained_dict['model_state_dict'].items():
            if k in model_dict and 'backbone.fc1' not in k and 'fc2' not in k.split('.')[0]:
                temp_pretrained_dict[k] = v
        model_dict.update(temp_pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

    edge_feature = torch.randn((5,512,6)) # [subject+object , xyz center_xyz, volume, etc..., 1 ]

    model_dict = model.state_dict()
    print(model_dict)
    # sys.exit()
    embedding = model(edge_feature)

    print(embedding.size())

    sys.exit()

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader
    from matplotlib import cm
    import numpy as np
    import matplotlib.pyplot as plt
    test_predictions = []
    test_embeddings = torch.zeros((0, 128), dtype=torch.float32)

    temp_num_index = [i for i in range(0,5)]
    for i in range(0,100):
        edge_feature = torch.randn((5, 512, 6))  # [subject+object , xyz center_xyz, volume, etc..., 1 ]
        embedding = model(edge_feature)
        test_embeddings = torch.cat((test_embeddings, embedding.detach().cpu()), 0)
        test_predictions.extend(temp_num_index)

    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(np.array(test_embeddings))
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 5

    print('tsne_proj : \n' ,tsne_proj)

    print('tsne_proj : \n' ,tsne_proj.shape)
    # test_predictions
    print('test_predictions : \n',test_predictions)
    # sys.exit()
    test_predictions = np.array(test_predictions)
    for lab in range(num_categories):
        indices = test_predictions == lab
        print('indeices = ',indices)
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
