from torch import Tensor


def zeros_feat_vector_check(x: Tensor):
    total_size = x.shape[0]
    total_all_zeros = total_size - x.count_nonzero(dim=1).count_nonzero().item()
    return {'allZeroFeatureVec_Count': total_all_zeros, 'allZeroFeatureVec_Percent': total_all_zeros / total_size}
