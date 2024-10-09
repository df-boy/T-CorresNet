import torch
from torch import nn

import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import Block, DecoderBlock, get_knn_index
from .build import MODELS
from torch.autograd import Variable
from .dgcnn_group import DGCNN
import numpy as np


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc


def fps_index(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    return fps_idx



def sample_one_sphere(num_points):
    out_points = torch.randn(num_points, 3).cuda()
    out_points = F.normalize(out_points, p=2, dim=1)
    return out_points


def sample_one_gaussian_sphere(num_points):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.arccos(np.random.uniform(-1, 1, num_points))
    u = np.random.uniform(0, 1, num_points)

    r = (1. * u ** (1 / 3))  # Reshape to column vector

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    pts = np.column_stack((x, y, z))
    pts = torch.tensor(pts, dtype=torch.float32).cuda()
    return pts

def sample_sphere(bsz, num_points):
    return torch.stack([sample_one_sphere(num_points) for _ in range(bsz)], dim=0)


def sample_gaussian_sphere(bsz, num_points):
     return torch.stack([sample_one_gaussian_sphere(num_points) for _ in range(bsz)], dim=0)
 

class STN(nn.Module):
    def __init__(self, k=64):
        super(STN, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.k = k

    def forward(self, x):
        # batch_size point_num k
        x = x.transpose(1, 2)
        # batch_size k point_num
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def corres_graph_feature(x, knn_index, x_q=None):
    #x: bs, np, c, knn_index: bs*k*np
    k = 8
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    return feature  # b k np c


class CorresAttention(nn.Module):
    def __init__(self, num_heads=4):
        super().__init__()
        self.k = 8
        self.corresAttention = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(3, 1, kernel_size=1, bias=False),
            nn.LayerNorm([1, 512]),
            nn.GELU(),
            nn.Conv1d(1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, u, x):
        # u: N x 1024 x 3, x: N x 512 x 3, index: N * 1024 * k
        B, N, C = u.shape
        idx = get_knn_index(u.transpose(1, 2), x.transpose(1, 2))
        x_f = corres_graph_feature(x, idx, u)
        x_f = torch.sum(F.softmax(x_f, dim=1), dim=1).squeeze(1)
        x_f = x_f.reshape(B, -1, C)
        u_f, _ = self.corresAttention(u, x_f, x_f)
        u_f = u_f.view(B, N, C).permute(0, 2, 1)
        u_f = self.conv(u_f).permute(0, 2, 1).squeeze(-1)
        return u_f


class sphericalTemplateGenerator(nn.Module):
    def __init__(self, in_scale, out_scale, depth=1, radius=10.0):
        super().__init__()
        self.radius = radius
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.ga = DGCNN()
        self.linear_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )
        self.u_linear_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )
        self.position_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 384),
        )
        self.u_position_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 384),
        )
        self.feature_upsample = nn.Sequential(
            nn.Conv1d(384, 1024, 1),
            nn.GroupNorm(4, 1024),
            nn.GELU(),
            nn.Conv1d(1024, 1024, 1),
        )
        self.input_ga = DGCNN()
        self.template_generator = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512 * 3)
        )
        self.encoder = nn.ModuleList([
            Block(dim=384, num_heads=4, mlp_ratio=2., drop=0., attn_drop=0.)
            for i in range(6)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(dim=384, num_heads=4, mlp_ratio=2., drop=0., attn_drop=0.)
            for i in range(8)
        ])
        self.mlp = nn.Sequential(
            nn.Conv1d(1024 + 3, 384, 1),
            nn.GroupNorm(4, 384),
            nn.GELU(),
            nn.Conv1d(384, 384, 1)
        )
        self.query_mlp = nn.Sequential(
            Mlp(in_features=384 + 3, out_features=384),
        )
        self.vote = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.corresAttention = CorresAttention()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        # sample n x 2048 x 3 unit sphere
        unit_sphere = sample_sphere(xyz.size()[0], self.in_scale)
        # graph-attention extract feature
        x_c, x_f = self.input_ga(xyz, [512, 256])
        u_c, u_f = self.ga(unit_sphere, [512, 256])
        # position embeding
        # x1_f = self.linear_proj(x1_f) + self.position_embed(x1)
        u_f = self.u_linear_proj(u_f) + self.u_position_embed(u_c)
        x_f =  self.linear_proj(x_f) + self.position_embed(x_c)
        # knn_index for graph-transformer encoder
        knn_index = get_knn_index(x_c.transpose(1, 2))
        for i, layer in enumerate(self.encoder):
            if i == 0:
                x_f = layer(x_f + u_f, knn_index)
            else:
                x_f = layer(x_f + u_f)
        # N x 256 x 384
        # N x 256 x 1024 global feature
        # decorrespondence matrix computing
        # maxpooling global feature to generate global template and template feature
        global_uf = torch.max(self.feature_upsample(x_f.transpose(1, 2)), dim=-1)[0]
        # N x 512 x 3
        u_template = self.template_generator(global_uf).reshape(xyz.size()[0], -1, 3) 
        # select k most input-related template to be replaced with k=64 points fpsed from x
        decorresU = self.corresAttention(u_template, xyz)
        u_index = torch.topk(decorresU, k=256, dim=-1, largest=True, sorted=False).indices.unsqueeze(-1)
        u_pick = torch.gather(u_template, 1, u_index.expand(-1, -1, 3))
        u_query = torch.cat([u_pick, fps(xyz, 256 + 128)], dim=1)
        score = self.vote(u_query)
        score_index = torch.argsort(score, descending=True, dim=1)
        u_query = torch.gather(u_query, 1, score_index[:,:512].expand(-1, -1, 3))
        # transformer decoder to transform global template to the detailed complete model
        self_denoised_length = 0
        self_knn_index = get_knn_index(u_query.transpose(1, 2).contiguous())
        cross_knn_index = get_knn_index(u_query.transpose(1, 2).contiguous(), x_c.transpose(1, 2).contiguous())
        query = torch.cat([
            global_uf.unsqueeze(1).expand(-1, u_query.size(1), -1),
            u_query], dim=-1)
        query = self.mlp(query.transpose(1, 2)).transpose(1, 2)
        value = torch.cat([
            x_f,
            sample_sphere(x_f.size(0), x_f.size(1)),
        ], dim=-1)
        value = self.query_mlp(value)
        for i, layer in enumerate(self.decoder):
            if i <= 2:
                query = layer(query, value, self_knn_index, cross_knn_index)
            else:
                query = layer(query, value, self_knn_index)
        return u_query, query, self_denoised_length, u_template


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc
    

@MODELS.register_module()
class TCorresNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.BaseModel = sphericalTemplateGenerator(in_scale=2048, out_scale=960, depth=3)
        self.reduce_map = nn.Linear(384 + 1024 + 3, 512)
        self.decode_head = SimpleRebuildFCLayer(512 * 2, step=32)
        self.build_loss_func()
        self.feature_upsample = nn.Sequential(
            nn.Conv1d(384, 1024, 1),
            nn.GroupNorm(4, 1024),
            nn.GELU(),
            nn.Conv1d(1024, 1024, 1)
        )

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, epoch):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[-1], gt)
        return loss_coarse, loss_fine
    
    def forward(self, x):
        x, x_f, self_denoise_length, u_template = self.BaseModel(x)
        B, N, C = x_f.shape
        global_feature = self.feature_upsample(x_f.transpose(1, 2)).transpose(1, 2)
        global_f = torch.max(global_feature, dim=1)[0]
        rebuild_f = torch.cat([global_f.unsqueeze(-2).expand(-1, N, -1),
                               x_f,
                               x], dim=-1)
        rebuild_f = self.reduce_map(rebuild_f)
        xyz_bias = self.decode_head(rebuild_f)
        x_fine = (xyz_bias + x.unsqueeze(-2))
        x_fine = x_fine.reshape(B, -1, 3).contiguous()
        assert x_fine.size(1) == 16384
        return (x, u_template, x_fine)