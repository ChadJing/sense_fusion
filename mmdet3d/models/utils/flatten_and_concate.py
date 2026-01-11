import torch

def flatten_and_concat_features(feat_maps):
    """
    Args:
        feat_maps: list of 3 feature maps, each with shape [bn, c, h, w]
    
    Returns:
        Tensor with shape [L, bs*num_cams, c]
    """
    # 存储展平后的特征
    flattened_features = []
    
    for feat in feat_maps:
        # 获取当前特征的维度
        bn, c, h, w = feat.shape
        
        # 展平空间维度：[num_cams*bs, c, h, w] -> [num_cams*bs, c, h*w]
        feat_flattened = feat.flatten(2)
        
        # 转置维度：[num_cams*bs, c, h*w] -> [h*w, num_cams*bs, c]
        feat_transposed = feat_flattened.permute(2, 0, 1)
        
        flattened_features.append(feat_transposed)
    
    # 在第一个维度（L维度）拼接所有特征
    result = torch.cat(flattened_features, dim=0)
    
    return result
