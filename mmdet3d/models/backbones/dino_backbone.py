import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
import timm

import torch.nn.functional as F
import torch
import os
from mmcv.runner import load_checkpoint
import logging

logger = logging.getLogger(__name__)

__all__ = ['DINOBackbone', 'DINOWithFPN']

@BACKBONES.register_module()
class DINOBackbone(BaseModule):
    def __init__(self,
                 model_name='vit_base_patch16_224',
                 target_size=(32,88),
                 pretrained=True,
                 freeze=True,
                 checkpoint_path=None,
                 dynamic_img_size=True,
                 output_dim=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self._is_patched = False
        
        self.model = timm.create_model(
            model_name, 
            pretrained=False,
            img_size=224,
            dynamic_img_size=dynamic_img_size,
            num_classes=0  # 移除分类头
        )

        # 获取特征维度
        if 'base' in model_name:
            self.feature_dim = 768
        elif 'small' in model_name:
            self.feature_dim = 384
        elif 'large' in model_name:
            self.feature_dim = 1024
        else:
            self.feature_dim = 768
        self.freeze=freeze
        self.model_name=model_name
        self.checkpoint_path=checkpoint_path
        if pretrained:
            self.load_pretrained_weights()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad=False
            self.model.eval()
        if output_dim is not None:
            self.output_dim=output_dim
        else:
            self.output_dim=self.feature_dim

        #定义一个上采样层，保证特征尺寸是想要的
        self.upsample_layer = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
            nn.Conv2d(self.feature_dim // 2, self.output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 初始化上采样层权重
        for m in self.upsample_layer:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  

    def load_pretrained_weights(self):
        """加载预训练权重"""
        # 优先使用本地权重文件
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Loading DINO weights from local file: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # 处理不同的权重格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # 适配timm模型的关键字
            new_state_dict = {}
            for k, v in state_dict.items():
                # 移除可能的prefix
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('backbone.'):
                    k = k[9:]
                # 适配timm的key命名
                if k.startswith('patch_embed.'):
                    k = k.replace('patch_embed.', 'patch_embed.')
                elif k.startswith('blocks.'):
                    k = k.replace('blocks.', 'blocks.')
                new_state_dict[k] = v
            # 处理positional encoding以适应动态尺寸
            new_state_dict = self.adapt_positional_encoding(new_state_dict)
                
            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading DINO weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading DINO weights: {unexpected_keys}")
                
        else:
            # 如果没有本地权重，尝试从timm下载（可能会失败）
            try:
                logger.info(f"Loading DINO weights from timm for {self.model_name}")
                model_with_pretrained = timm.create_model(
                    self.model_name, 
                    pretrained=True,
                    num_classes=0
                )
                self.model.load_state_dict(model_with_pretrained.state_dict())
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
                logger.info("Using randomly initialized weights")
    def adapt_positional_encoding(self, state_dict):
        """适配positional encoding以支持动态尺寸"""
        # 查找pos_embed相关的key
        pos_embed_keys = [k for k in state_dict.keys() if 'pos_embed' in k]
        import math
        for key in pos_embed_keys:
            if key in state_dict:
                original_pos_embed = state_dict[key]
                original_size = int(math.sqrt(original_pos_embed.shape[1] - 1))  # 减去cls token
                
                print(f"Original positional encoding size: {original_size}x{original_size}")
                print(f"Positional encoding shape: {original_pos_embed.shape}")
                
                # 对于动态尺寸，我们保持原始的pos_embed
                # 在forward时，timm会自动进行插值
                # 这里我们只需要确保权重被正确加载
                
        return state_dict
    def vit_features_to_spatial(self, patch_tokens, batch_size, image_size, patch_size=16):
        """Convert ViT patch tokens to spatial feature map"""
        if patch_tokens is None:
            raise ValueError("patch_tokens is None")
        h = image_size[0] // patch_size
        w = image_size[1] // patch_size
        if patch_tokens.dim() == 3:
            expected_patches = h * w
            actual_patches = patch_tokens.shape[1]
            
            if actual_patches != expected_patches:
                if actual_patches == expected_patches + 1:
                    patch_tokens = patch_tokens[:, 1:]
                else:
                    pass
            
            features = patch_tokens.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        else:
            features = patch_tokens.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        return features
    def patch_antialias_interpolate(self):
        if self._is_patched:
            return
        """修补interpolate函数"""
        import torch.nn.functional as F
        from functools import wraps
        
        original_interpolate = F.interpolate
        
        @wraps(original_interpolate)
        def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', 
                               align_corners=None, recompute_scale_factor=None, **kwargs):
            if 'antialias' in kwargs:
                kwargs.pop('antialias')
            return original_interpolate(input, size, scale_factor, mode, align_corners, 
                                      recompute_scale_factor, **kwargs)
        
        F.interpolate = patched_interpolate
        torch.nn.functional.interpolate = patched_interpolate
        self._is_patched = True
        print("✅ Interpolate function patched (only once)")
    def forward(self, x):
        self.patch_antialias_interpolate()
        batch_size = x.shape[0]
        img_size=(x.shape[2],x.shape[3])
        
        features = self.model.forward_features(x)
        
        if isinstance(features, dict):
            if 'cls_token' in features:
                cls_token = features['cls_token']
            patch_tokens = features['x']
        else:
            if len(features.shape) == 3:
                patch_tokens = features
                cls_token = features[:, 0]
            else:
                cls_token = features
                patch_tokens = None
        if patch_tokens is None:
            raise ValueError("DINO backbone did not return patch tokens")
            
        # 转换为空间特征图,并且直接返回；处理成为图像形式的特征2d
        spatial_features = self.vit_features_to_spatial(patch_tokens, batch_size, img_size)
        features_upsampled = self.upsample_layer(spatial_features)
        return features_upsampled



@BACKBONES.register_module()
class DINOWithFPN(BaseModule):
    """DINO backbone + FPN for multi-scale features"""
    def __init__(self,
                 model_name='vit_base_patch16_224',
                 pretrained=True,
                 freeze=True,
                 out_channels=256,
                 fpn_levels=[2, 3, 4],
                 target_feat_size=(32,88),
                 checkpoint_path=None, 
                 dynamic_img_size=True, 
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 保证只修复一次
        self._is_patched=False

        self.backbone = DINOBackbone(
            model_name, 
            pretrained, 
            freeze,checkpoint_path=checkpoint_path,
            dynamic_img_size=dynamic_img_size
        )
        self.fpn_levels = fpn_levels
        self.out_channels = out_channels
        self.target_feat_size=target_feat_size
        
        # 创建FPN层
        self.fpn_convs = nn.ModuleDict()
        
        for level in fpn_levels:
            # 1x1卷积调整通道数
            self.fpn_convs[f'conv1x1_{level}'] = nn.Conv2d(
                self.backbone.feature_dim, out_channels, kernel_size=1, stride=1, padding=0
            )
            
            # 3x3卷积
            self.fpn_convs[f'conv3x3_{level}'] = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        # 增加上采样层，
        self.upsample_layers= nn.ModuleDict()
        for level in fpn_levels:
            self.upsample_layers[f'upsample_{level}'] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(size=target_feat_size, mode='bilinear', align_corners=False),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        # 初始化FPN权重
        for m in self.fpn_convs.values():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.upsample_layers.values():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def patch_antialias_interpolate(self):
        if self._is_patched:
            return
        """修补interpolate函数"""
        import torch.nn.functional as F
        from functools import wraps
        
        original_interpolate = F.interpolate
        
        @wraps(original_interpolate)
        def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', 
                               align_corners=None, recompute_scale_factor=None, **kwargs):
            if 'antialias' in kwargs:
                kwargs.pop('antialias')
            return original_interpolate(input, size, scale_factor, mode, align_corners, 
                                      recompute_scale_factor, **kwargs)
        
        F.interpolate = patched_interpolate
        torch.nn.functional.interpolate = patched_interpolate
        self._is_patched = True
        print("✅ Interpolate function patched (only once)")


        
    def forward(self, x):
        self.patch_antialias_interpolate()

        # 获取DINO特征
        spatial_features = self.backbone(x)

        print(f"Spatial features shape: {spatial_features.shape}")

        # FPN处理
        fpn_outputs = []
        
        for level in self.fpn_levels:
            # 下采样到对应层级
            scale_factor = 2 ** (level - 2)
            target_h = spatial_features.shape[2] // scale_factor
            target_w = spatial_features.shape[3] // scale_factor
            
            if scale_factor == 1:
                resized_features = spatial_features
            else:
                resized_features = F.interpolate(
                    spatial_features, size=(target_h,target_w), mode='bilinear', align_corners=False
                )
                
            # 1x1卷积调整通道数
            features = self.fpn_convs[f'conv1x1_{level}'](resized_features)
            # 3x3卷积
            features = self.fpn_convs[f'conv3x3_{level}'](features)
            
            features_upsampled = self.upsample_layers[f'upsample_{level}'](features)
            
            fpn_outputs.append(features_upsampled)
            
        return tuple(fpn_outputs)