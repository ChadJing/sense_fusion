# 复用信息
input_modality_dict= {       
    'use_camera': True,
    'use_lidar': True,
    'use_radar': False,
    'use_map': False,
    'use_external': False
    }
map_classes_list=[
    'drivable_area', 'ped_crossing', 'walkway',
    'stop_line', 'carpark_area', 'divider'
]
object_classes_list = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
test_val_eval_pipeline = [
                # 1. 加载多视角图像
                {
                    'type': 'LoadMultiViewImageFromFiles',
                    'to_float32': True
                },
                
                # 2. 加载LiDAR点云
                {
                    'type': 'LoadPointsFromFile',
                    'coord_type': 'LIDAR',
                    'load_dim': 5,
                    'use_dim': 5,
                    'reduce_beams': 32,
                    'load_augmented': None
                },
                
                # 3. 加载多帧点云
                {
                    'type': 'LoadPointsFromMultiSweeps',
                    'sweeps_num': 9,
                    'load_dim': 5,
                    'use_dim': 5,
                    'pad_empty_sweeps': True,
                    'remove_close': True,
                    'reduce_beams': 32,
                    'load_augmented': None
                },
                
                # 4. 加载Radar点云  TODO: 确定是否需要加载Radar 点云
                {
                    'type': 'LoadRadarPointsMultiSweeps',
                    'sweeps_num': 6,
                    'load_dim': 18,
                    'use_dim': list(range(1, 57)),  # 使用1-56维度
                    'max_num': 2500,
                    'normalize': False,
                    'compensate_velocity': True,
                    'filtering': 'none'
                },
                
                # 5. 加载3D标注
                {
                    'type': 'LoadAnnotations3D',
                    'with_bbox_3d': True,
                    'with_label_3d': True,
                    'with_attr_label': False
                },
                
                # 6. 图像数据增强
                {
                    'type': 'ImageAug3D',
                    'is_train': False,
                    'final_dim': [256, 704],
                    'resize_lim': [0.48, 0.48],
                    'rot_lim': [0.0, 0.0],
                    'bot_pct_lim': [0.0, 0.0],
                    'rand_flip': False
                },
                
                # 7. 全局3D变换
                {
                    'type': 'GlobalRotScaleTrans',
                    'is_train': False,
                    'resize_lim': [1.0, 1.0],
                    'rot_lim': [0.0, 0.0],
                    'trans_lim': 0.0
                },
                
                # 8. 点云范围过滤
                {
                    'type': 'PointsRangeFilter',
                    'point_cloud_range': point_cloud_range
                },
                
                # 9. 图像归一化
                {
                    'type': 'ImageNormalize',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                },
                
                # 10. 数据格式化
                {
                    'type': 'DefaultFormatBundle3D',
                    'classes': [
                        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                    ]
                },
                
                # 11. 数据收集
                {
                    'type': 'Collect3D',
                    'keys': ['img', 'points', 'radar', 'gt_bboxes_3d', 'gt_labels_3d'],
                    'meta_keys': [
                        'camera_intrinsics', 'camera2ego', 'lidar2ego',
                        'lidar2camera', 'lidar2image', 'camera2lidar',
                        'img_aug_matrix', 'lidar_aug_matrix'
                    ]
                },
                
                # 12. 生成GT深度图
                {
                    'type': 'GTDepth', 
                    'keyframe_only': True
                }
            ]

config_dict = {
    # ============ 数据增强配置 ============
    'augment2d': {
        'gridmask': {
            'fixed_prob': True,
            'prob': 0.0
        },
        'resize': [[0.38, 0.55], [0.48, 0.48]],
        'rotate': [-5.4, 5.4]
    },
    
    'augment3d': {
        'rotate': [-0.78539816, 0.78539816],  # 弧度制：约 -45° 到 45°
        'scale': [0.9, 1.1],
        'translate': 0.5
    },
    
    # ============ 训练配置 ============
    'checkpoint_config': {
        'interval': 1,
        'max_keep_ckpts': 1
    },
    
    'cudnn_benchmark': False,
    'deterministic': False,
    
    # ============ 数据配置 ============
    'data': {
        'num_gpus': 1,
        'samples_per_gpu': 1,
        'workers_per_gpu': 2,
        
        # 测试集配置
        'test': {
            'type': 'NuScenesDataset',
            'dataset_root': 'data/nuscenes/',
            'ann_file': 'data/nuscenes/nuscenes_infos_val.pkl',
            'test_mode': True,
            'box_type_3d': 'LiDAR',
            
            # 传感器模态
            'modality': input_modality_dict,
            
            # 类别定义
            'object_classes': object_classes_list,
            
            'map_classes': map_classes_list,
            
            # 数据处理流水线
            'pipeline': test_val_eval_pipeline
        },
        
        # 训练集配置
        'train': {
            'type': 'CBGSDataset',
            'dataset': {
                'type': 'NuScenesDataset',
                'dataset_root': 'data/nuscenes/',
                'ann_file': 'data/nuscenes/nuscenes_infos_train.pkl',
                'test_mode': False,
                'use_valid_flag': True,
                'box_type_3d': 'LiDAR',
                
                # 传感器模态
                'modality': input_modality_dict,
                
                # 类别定义
                'object_classes': object_classes_list,
                
                'map_classes': map_classes_list,
                
                # 训练数据处理流水线（更复杂，包含数据增强）
                'pipeline': [
                    # 1-4. 数据加载（与测试集相同）
                    {'type': 'LoadMultiViewImageFromFiles', 'to_float32': True},
                    {
                        'type': 'LoadPointsFromFile',
                        'coord_type': 'LIDAR',
                        'load_dim': 5,
                        'use_dim': 5,
                        'reduce_beams': 32,
                        'load_augmented': None
                    },
                    {
                        'type': 'LoadPointsFromMultiSweeps',
                        'sweeps_num': 0,  # 训练时使用0帧
                        'load_dim': 5,
                        'use_dim': 5,
                        'pad_empty_sweeps': True,
                        'remove_close': True,
                        'reduce_beams': 32,
                        'load_augmented': None
                    },
                    { #TODO: 确定是否需要加载Radar 点云
                        'type': 'LoadRadarPointsMultiSweeps',
                        'sweeps_num': 6,
                        'load_dim': 18,
                        'use_dim': list(range(1, 57)),
                        'max_num': 2500,
                        'normalize': False,
                        'compensate_velocity': True,
                        'filtering': 'none'
                    },
                    
                    # 5. 加载3D标注
                    {
                        'type': 'LoadAnnotations3D',
                        'with_bbox_3d': True,
                        'with_label_3d': True,
                        'with_attr_label': False
                    },
                    
                    # 6. 数据增强：物体粘贴（增强小样本）
                    {
                        'type': 'ObjectPaste',
                        'stop_epoch': -1,
                        'db_sampler': {
                            'rate': 1.0,
                            'dataset_root': 'data/nuscenes/',
                            'info_path': 'data/nuscenes/nuscenes_dbinfos_train.pkl',
                            'points_loader': {
                                'type': 'LoadPointsFromFile',
                                'coord_type': 'LIDAR',
                                'load_dim': 5,
                                'use_dim': 5,
                                'reduce_beams': 32
                            },

                            'prepare': {
                                'filter_by_difficulty': [-1],
                                'filter_by_min_points': {
                                    'car': 5, 'truck': 5, 'bus': 5,
                                    'trailer': 5, 'construction_vehicle': 5,
                                    'motorcycle': 5, 'bicycle': 5,
                                    'pedestrian': 5, 'barrier': 5,
                                    'traffic_cone': 5
                                }
                            },
                            'sample_groups': {
                                'car': 2, 'truck': 3, 'bus': 4,
                                'trailer': 6, 'construction_vehicle': 7,
                                'motorcycle': 6, 'bicycle': 6,
                                'pedestrian': 2, 'barrier': 2,
                                'traffic_cone': 2
                            },
                            'classes': [
                                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                            ]
                        }
                    },
                    
                    # 7. 图像数据增强（训练时启用）
                    {
                        'type': 'ImageAug3D',
                        'is_train': True,
                        'final_dim': [256, 704],
                        'resize_lim': [0.38, 0.55],
                        'rot_lim': [-5.4, 5.4],
                        'bot_pct_lim': [0.0, 0.0],
                        'rand_flip': True
                    },
                    
                    # 8. 全局3D变换（训练时启用）
                    {
                        'type': 'GlobalRotScaleTrans',
                        'is_train': True,
                        'resize_lim': [0.9, 1.1],
                        'rot_lim': [-0.78539816, 0.78539816],
                        'trans_lim': 0.5
                    },
                    
                    # 9. 随机翻转
                    {'type': 'RandomFlip3D'},
                    
                    # 10-12. 过滤操作 todo 为什么有两个PointsRangeFilter
                    {
                        'type': 'PointsRangeFilter',
                        'point_cloud_range': point_cloud_range
                    },
                    {
                        'type': 'ObjectRangeFilter',
                        'point_cloud_range': point_cloud_range
                    },
                    {
                        'type': 'ObjectNameFilter',
                        'classes': [
                            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                        ]
                    },
                    
                    # 13. 图像归一化
                    {
                        'type': 'ImageNormalize',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    },
                    
                    # 14. GridMask增强
                    {
                        'type': 'GridMask',
                        'prob': 0.0,
                        'fixed_prob': True,
                        'max_epoch': 100,
                        'mode': 1,
                        'ratio': 0.5,
                        'rotate': 1,
                        'offset': False,
                        'use_h': True,
                        'use_w': True
                    },
                    
                    # 15. 点云打乱
                    {'type': 'PointShuffle'},
                    
                    # 16-18. 数据格式化、收集、GT深度（与测试集相同）
                    {
                        'type': 'DefaultFormatBundle3D',
                        'classes': [
                            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                        ]
                    },
                    {
                        'type': 'Collect3D',
                        'keys': ['img', 'points', 'radar', 'gt_bboxes_3d', 'gt_labels_3d'],
                        'meta_keys': [
                            'camera_intrinsics', 'camera2ego', 'lidar2ego',
                            'lidar2camera', 'lidar2image', 'camera2lidar',
                            'img_aug_matrix', 'lidar_aug_matrix'
                        ]
                    },
                    {'type': 'GTDepth', 'keyframe_only': True}
                ]
            }
        },
        
        # 验证集配置（与测试集基本相同）
        'val': {
            'type': 'NuScenesDataset',
            'dataset_root': 'data/nuscenes/',
            'ann_file': 'data/nuscenes/nuscenes_infos_val.pkl',
            'test_mode': False,
            'box_type_3d': 'LiDAR',
            'modality': input_modality_dict,
            'object_classes': object_classes_list,
            'map_classes': map_classes_list,
            'pipeline': test_val_eval_pipeline
        }
    },
    
    # ============ 数据集基本信息 ============
    'dataset_root': 'data/nuscenes/',
    'dataset_type': 'NuScenesDataset',
    
    'object_classes': object_classes_list,
    
    'map_classes': map_classes_list,
    
    'point_cloud_range': point_cloud_range,
    
    # ============ 输入模态配置 ============
    'input_modality': input_modality_dict,
    
    'image_size': [256, 704],
    
    # ============ Radar配置 ============
    'radar_sweeps': 6,
    'radar_max_points': 2500,
    'radar_normalize': False,
    'radar_compensate_velocity': True,
    'radar_filtering': 'none',
    'radar_jitter': 0,
    'radar_use_dims': list(range(1, 57)),
    'radar_voxel_size': [0.8, 0.8, 8],
    
    # ============ LiDAR配置 ============
    'load_dim': 5,
    'use_dim': 5,
    'reduce_beams': 32,
    'voxel_size': [0.075, 0.075, 0.2],
    
    # ============ 模型配置 ============
    'model': {
        'type': 'BEVFusion',
        
        # 编码器部分：相机和LiDAR
        'encoders': {
            # 相机编码器
            'camera': {
                'backbone': {
                    'type': 'SwinTransformer',
                    'embed_dims': 96,
                    'depths': [2, 2, 6, 2],
                    'num_heads': [3, 6, 12, 24],
                    'convert_weights': True,
                    'window_size': 7,
                    'mlp_ratio': 4,
                    'qkv_bias': True,
                    'qk_scale': None,
                    'drop_rate': 0.0,
                    'attn_drop_rate': 0.0,
                    'drop_path_rate': 0.2,
                    'patch_norm': True,
                    'out_indices': [1, 2, 3],
                    'with_cp': False,
                    'init_cfg': {
                        'type': 'Pretrained',
                        'checkpoint': 'pretrained/swint-nuimages-pretrained.pth'
                    }
                },
                'neck': {
                    'type': 'GeneralizedLSSFPN',
                    'in_channels': [192, 384, 768],
                    'out_channels': 256,
                    'num_outs': 3,
                    'start_level': 0,
                    'norm_cfg': {'type': 'BN2d', 'requires_grad': True},
                    'act_cfg': {'type': 'ReLU', 'inplace': True},
                    'upsample_cfg': {'mode': 'bilinear', 'align_corners': False}
                },
                'vtransform': {
                    'type': 'BEVTransformer',
                    'num_cams': 6,
                    'num_levels': 2,
                    'embed_dims': 256,
                    'bev_h': 180,
                    'bev_w': 180,
                    'num_Z_anchors': 4,
                    'out_dims': 80,
                }
            },
            
            # LiDAR编码器
            'lidar': {
                'voxelize': {
                    'max_num_points': 10,
                    'max_voxels': [120000, 160000],
                    'point_cloud_range': point_cloud_range,
                    'voxel_size': [0.075, 0.075, 0.2]
                },
                'backbone': {
                    'type': 'SparseEncoder',
                    'in_channels': 5,
                    'sparse_shape': [1440, 1440, 41],
                    'output_channels': 128,
                    'encoder_channels': [
                        [16, 16, 32],
                        [32, 32, 64],
                        [64, 64, 128],
                        [128, 128]
                    ],
                    'encoder_paddings': [
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, [1, 1, 0]],
                        [0, 0]
                    ],
                    'block_type': 'basicblock',
                    'order': ['conv', 'norm', 'act']
                }
            }
        },
        
        # 特征融合器
        'fuser': {
            'type': 'ConvFuser',
            'in_channels': [80, 256],  # 相机特征 + LiDAR特征
            'out_channels': 256
        },
        
        # 解码器（BEV特征提取）
        'decoder': {
            'backbone': {
                'type': 'SECOND',
                'in_channels': 256,
                'out_channels': [128, 256],
                'layer_nums': [5, 5],
                'layer_strides': [1, 2],
                'norm_cfg': {'type': 'BN', 'eps': 0.001, 'momentum': 0.01},
                'conv_cfg': {'type': 'Conv2d', 'bias': False}
            },
            'neck': {
                'type': 'SECONDFPN',
                'in_channels': [128, 256],
                'out_channels': [256, 256],
                'upsample_strides': [1, 2],
                'use_conv_for_no_stride': True,
                'norm_cfg': {'type': 'BN', 'eps': 0.001, 'momentum': 0.01},
                'upsample_cfg': {'type': 'deconv', 'bias': False}
            }
        },
        
        # 检测头
        'heads': {
            'object': {
                'type': 'TransFusionHead',
                'num_classes': 10,
                'in_channels': 512,
                'hidden_channel': 128,
                'ffn_channel': 256,
                'num_heads': 8,
                'num_decoder_layers': 1,
                'num_proposals': 200,
                'nms_kernel_size': 3,
                'activation': 'relu',
                'dropout': 0.1,
                'bn_momentum': 0.1,
                'auxiliary': True,
                
                # 回归头配置
                'common_heads': {
                    'center': [2, 2],
                    'height': [1, 2],
                    'dim': [3, 2],
                    'rot': [2, 2],
                    'vel': [2, 2]
                },
                
                # 边界框编码器
                'bbox_coder': {
                    'type': 'TransFusionBBoxCoder',
                    'code_size': 10,
                    'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    'score_threshold': 0.0,
                    'pc_range': [-54.0, -54.0],
                    'voxel_size': [0.075, 0.075],
                    'out_size_factor': 8
                },
                
                # 损失函数配置
                'loss_heatmap': {
                    'type': 'GaussianFocalLoss',
                    'loss_weight': 1.0,
                    'reduction': 'mean'
                },
                'loss_bbox': {
                    'type': 'L1Loss',
                    'loss_weight': 0.25,
                    'reduction': 'mean'
                },
                'loss_cls': {
                    'type': 'FocalLoss',
                    'use_sigmoid': True,
                    'gamma': 2.0,
                    'alpha': 0.25,
                    'loss_weight': 1.0,
                    'reduction': 'mean'
                },
                
                # 训练配置
                'train_cfg': {
                    'dataset': 'nuScenes',
                    'assigner': {
                        'type': 'HungarianAssigner3D',
                        'cls_cost': {
                            'type': 'FocalLossCost',
                            'weight': 0.15,
                            'alpha': 0.25,
                            'gamma': 2.0
                        },
                        'reg_cost': {
                            'type': 'BBoxBEVL1Cost',
                            'weight': 0.25
                        },
                        'iou_cost': {
                            'type': 'IoU3DCost',
                            'weight': 0.25
                        },
                        'iou_calculator': {
                            'type': 'BboxOverlaps3D',
                            'coordinate': 'lidar'
                        }
                    },
                    'gaussian_overlap': 0.1,
                    'min_radius': 2,
                    'grid_size': [1440, 1440, 41],
                    'voxel_size': [0.075, 0.075, 0.2],
                    'out_size_factor': 8,
                    'point_cloud_range': point_cloud_range,
                    'code_weights': [1.0] * 8 + [0.2, 0.2],
                    'pos_weight': -1
                },
                
                # 测试配置
                'test_cfg': {
                    'dataset': 'nuScenes',
                    'grid_size': [1440, 1440, 41],
                    'voxel_size': [0.075, 0.075],
                    'out_size_factor': 8,
                    'pc_range': [-54.0, -54.0],
                    'nms_type': None
                }
            },
            'map': None  # 不进行地图分割
        }
    },
    
    # ============ 优化器配置 ============
    'optimizer': {
        'type': 'AdamW',
        'lr': 0.0002,
        'weight_decay': 0.01
    },
    
    'optimizer_config': {
        'grad_clip': {
            'max_norm': 35,
            'norm_type': 2
        }
    },
    
    'lr_config': {
        'policy': 'CosineAnnealing',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_ratio': 0.33333333,
        'min_lr_ratio': 0.001
    },
    
    'momentum_config': {
        'policy': 'cyclic'
    },
    
    # ============ 训练运行配置 ============
    'runner': {
        'type': 'CustomEpochBasedRunner',
        'max_epochs': 100
    },
    
    'max_epochs': 100,
    'gt_paste_stop_epoch': -1,
    
    # ============ 评估配置 ============
    'evaluation': {
        'interval': 1,
        'pipeline': test_val_eval_pipeline
    },
    
    # ============ 日志配置 ============
    'log_config': {
        'interval': 50,
        'hooks': [
            {'type': 'TextLoggerHook'},
            {'type': 'TensorboardLoggerHook'}
        ]
    },
    
    # ============ 检查点与恢复 ============
    'load_from': 'pretrained/lidar-only-det.pth',
    # 'resume_from': 'results/epoch_5.pth',
    'resume_from': None,
    
    # ============ 混合精度训练 ============
    'fp16': {
        'loss_scale': {
            'growth_interval': 2000
        }
    },
    
    # ============ 环境配置 ============
    'seed': 0,
    'launcher': 'none',
    'local_rank': 0,
    'run_dir': 'results/'
}

# 使用示例
if __name__ == '__main__':
    from mmcv import Config
    cfg = Config(config_dict)
    print(f"数据集: {cfg.dataset_type}")
    print(f"模型: {cfg.model.type}")
    print(f"训练轮数: {cfg.max_epochs}")
    print(f"输入模态: 相机={cfg.input_modality.use_camera}, LiDAR={cfg.input_modality.use_lidar}")
    print(f"检测类别数: {len(cfg.object_classes)}")