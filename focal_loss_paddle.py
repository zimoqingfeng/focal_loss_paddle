# -*-coding utf-8 -*-
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# @Time : 2020/2/23 11:56 上午
# @Author : wubinghong
# @FileName: focal_loss_paddle.py.py

import paddle
import paddle.fluid as fluid
import numpy as np


def focal_loss(logit, label, class_dim, gamma=2.0, alpha=None, smooth=None):
    """Calculate focal loss

    Returns:
        weighted focal loss

    """

    if alpha is None:
        alpha = np.ones((class_dim, 1), dtype=float)
    else:
        alpha = np.reshape(np.array(alpha, dtype=float), (class_dim, 1))
    alpha = fluid.layers.create_parameter(shape=[class_dim, 1],
                                          dtype='float32',
                                          name='alpha',
                                          default_initializer=fluid.initializer.NumpyArrayInitializer(alpha))
    alpha.stop_gradient = True

    epsilon = 1e-10
    one_hot_key = fluid.layers.one_hot(input=label, depth=class_dim)
    alpha_matrix = fluid.layers.matmul(one_hot_key, alpha, transpose_x=False, transpose_y=False)
    alpha_raw = fluid.layers.squeeze(input=alpha_matrix, axes=[1])
    if smooth:
        one_hot_key = fluid.layers.label_smooth(label=one_hot_key, epsilon=smooth, dtype="float32")
    pt = fluid.layers.reduce_sum(one_hot_key * logit, dim=-1) + epsilon
    logpt = fluid.layers.log(pt)
    loss = -1.0 * fluid.layers.pow((1 - pt), gamma) * logpt * alpha_raw
    loss = fluid.layers.reduce_sum(loss)

    return loss
