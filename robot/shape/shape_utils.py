import numpy as np


def get_scale_and_center(points, percentile=99):
    dim = points.shape[1]
    scale = np.zeros([1, dim])
    center = np.zeros([1, dim])
    interval = [(100.0 - percentile) / 2, 100 - (100.0 - percentile) / 2]
    for d in range(dim):
        filtered_low_thre = np.percentile(points[:, d], interval[0])
        filtered_up_thre = np.percentile(points[:, d], interval[1])
        scale[0, d] = filtered_up_thre - filtered_low_thre
        center[0, d] = (filtered_up_thre + filtered_low_thre) / 2
    return (scale / 2).astype(np.float32), center.astype(np.float32)
