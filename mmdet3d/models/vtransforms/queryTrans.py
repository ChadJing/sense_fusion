import torch
import torch.nn as nn
from .spatial_cross_attention import SpatialCrossAttention
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
import numpy as np
__all__ = ['BEVTransformer']
from mmdet3d.models.builder import VTRANSFORMS

@VTRANSFORMS.register_module()
class BEVTransformer(nn.Module):
    """
    完整的BEV视角转换模块
    将多相机特征转换为BEV特征
    """
    
    def __init__(self, 
                 bev_h=100, 
                 bev_w=100,
                 embed_dims=256,
                 out_dims=80,
                 num_cams=6,
                 num_levels=4,
                 pc_range=None,
                 num_Z_anchors=4):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = bev_h * bev_w
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_Z_anchors = num_Z_anchors
        self.out_dims = out_dims
        
        if pc_range is None:
            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        else:
            self.pc_range = pc_range
        
        # 1. BEV查询（可学习参数）
        self.bev_embed = nn.Embedding(self.num_query, embed_dims)
        nn.init.normal_(self.bev_embed.weight, mean=0.0, std=0.02)
        
        # 2. BEV位置编码（几何位置）
        self.bev_pos_encoder = self._create_position_encoder()
        
        # 3. 空间交叉注意力
        self.spatial_cross_attention = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            pc_range=self.pc_range,
            batch_first=False,  # 注意：你的模块期望 [num_query, bs, embed_dims]
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=embed_dims,
                num_heads=8,
                num_levels=num_levels,
                num_points=8,
                batch_first=False
            )
        )
        
        # 4. 可选的BEV特征精炼层
        self.bev_refine = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, self.out_dims)
        )
    
    def _create_position_encoder(self):
        """创建BEV位置编码器"""
        return nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
    
    def get_bev_position_encoding(self, batch_size, device):
        """生成BEV位置编码"""
        # 生成归一化的2D网格坐标
        y_coords = torch.linspace(0, 1, self.bev_h, device=device)
        x_coords = torch.linspace(0, 1, self.bev_w, device=device)
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # grid_x: [bev_h, bev_w], grid_y: [bev_h, bev_w]
        
        # 拼接并展平
        pos_2d = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        
        # 通过MLP编码
        pos_encoding = self.bev_pos_encoder(pos_2d)  # [num_query, embed_dims]
        
        # 扩展到批次维度
        pos_encoding = pos_encoding.unsqueeze(1).expand(-1, batch_size, -1)
        
        return pos_encoding
    
    def get_bev_queries(self, batch_size, device):
        """获取BEV查询"""
        # 可学习的BEV查询
        bev_queries = self.bev_embed.weight  # [num_query, embed_dims]
        
        # 扩展到批次维度
        bev_queries = bev_queries.unsqueeze(1).expand(-1, batch_size, -1)
        
        return bev_queries
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z # 相当于在0.5-z-0.5区域内均匀采集n个点，然后归一化，扩展
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # (num_points_in_pillar,H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1) 
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  lidar2img, img_shape): # 本质上是去除不在相机视锥内的BEV query
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if lidar2img.is_cuda == False:
            lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4).此处是继承reference_points的device和dtype


        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0] # 将归一化的坐标映射回真实坐标系，其中reference_points.shape=[b,d,num_query,3]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5
        # 生成bev_mask，保证投影点位于相机视锥内
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps) # 此时的shape[D,B,num_cam,num_query,2],保留着所有点的信息

        reference_points_cam[..., 0] /= img_shape[1]
        reference_points_cam[..., 1] /= img_shape[0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # (num_cam, B, num_query, D, 2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # (num_cam, B, num_query, D)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask
    
    def forward(self, 
                multi_cam_features,  # (l,bn,c)
                lidar2img,           # 相机投影矩阵 [bs, num_cams, 4, 4]
                spatial_shapes,      # 多尺度特征图形状 [num_levels, 2]
                level_start_index,   # 各层起始索引 [num_levels]
                img_shape=None,
                img_metas=None):
        """
        前向传播
        
        参数:
            multi_cam_features: 多相机特征 tuple([num_cams, bs, C, H, W])
            lidar2img: LiDAR到图像的投影矩阵 [bs, num_cams, 4, 4]
            spatial_shapes: 特征金字塔形状 [num_levels, 2]
            level_start_index: 各层起始索引 [num_levels]
        """
        
        batch_size = lidar2img.size(0)
        device = multi_cam_features.device
        
        # 1. 准备BEV查询和位置编码
        bev_query = self.get_bev_queries(batch_size, device)  # [num_query, bs, embed_dims]
        bev_pos = self.get_bev_position_encoding(batch_size, device)  # [num_query, bs, embed_dims]
        
        
        # 这里假设multi_cam_features已经是展平的特征
        # 实际中需要根据spatial_shapes展平多尺度特征
        key = multi_cam_features #(L, bs*num_cams, embed_dims)
        value = multi_cam_features
        ref_3d = BEVTransformer.get_reference_points(
            H=self.bev_h, 
            W=self.bev_w, 
            Z=(self.pc_range[5]-self.pc_range[2]), 
            num_points_in_pillar=self.num_Z_anchors, 
            dim='3d', 
            bs=bev_query.size(1),  
            device=bev_query.device, 
            dtype=bev_query.dtype)

        # 3. 生成3D参考点并投影到2D图像
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, lidar2img, img_shape)

        # reference_points_cam: [num_cams, bs, num_query, D, 2]
        # bev_mask: [num_cams, bs, num_query, D]
        
        # 4. 执行空间交叉注意力
        bev_features = self.spatial_cross_attention(
            query=bev_query,           # [num_query, bs, embed_dims]
            key=key,                   # 多相机特征
            value=value,               # 多相机特征
            query_pos=bev_pos,         # BEV位置编码
            reference_points_cam=reference_points_cam,  # 投影后的2D点
            bev_mask=bev_mask,         # 可见性掩码
            spatial_shapes=spatial_shapes,      # 多尺度形状
            level_start_index=level_start_index # 层级起始索引
        )
        # bev_features: [num_query, bs, embed_dims]
        
        # 5. 可选：精炼BEV特征
        bev_features = self.bev_refine(bev_features)
        
        # 6. 重塑为BEV网格格式 [bs, C, bev_h, bev_w]
        bev_features = bev_features.permute(1, 2, 0)  # [bs, embed_dims, num_query]
        bev_features = bev_features.view(batch_size, self.out_dims, self.bev_h, self.bev_w)
        
        return bev_features
    
    