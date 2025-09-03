# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch


class XBM:
    def __init__(self, K, dim, device):
        self.K = K
        self.feats_1 = torch.zeros(self.K, dim).to(device)
        self.feats_2 = torch.zeros(self.K, dim).to(device)
        self.ptr = 0

    @property
    def is_full(self):
        return self.feats_2[-1, -1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats_1, self.feats_2
        else:
            return self.feats_1[:self.ptr], self.feats_2[:self.ptr]

    def enqueue_dequeue(self, feats_1, feats_2):
        q_size = len(feats_2)
        if self.ptr + q_size > self.K:
            self.feats_1[-q_size:] = feats_1
            self.feats_2[-q_size:] = feats_2
            self.ptr = 0
        else:
            self.feats_1[self.ptr: self.ptr + q_size] = feats_1
            self.feats_2[self.ptr: self.ptr + q_size] = feats_2
            self.ptr += q_size

class XBM_SSM:
    def __init__(self, K, dim1, dim2, device):
        self.K = K
        self.feats_1 = torch.zeros(self.K, dim1).to(device)
        self.feats_2 = torch.zeros(self.K, dim2).to(device)
        self.ptr = 0

    @property
    def is_full(self):
        return self.feats_2[-1, -1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats_1, self.feats_2
        else:
            return self.feats_1[:self.ptr], self.feats_2[:self.ptr]

    def enqueue_dequeue(self, feats_1, feats_2):
        q_size = len(feats_2)
        if self.ptr + q_size > self.K:
            self.feats_1[-q_size:] = feats_1
            self.feats_2[-q_size:] = feats_2
            self.ptr = 0
        else:
            self.feats_1[self.ptr: self.ptr + q_size] = feats_1
            self.feats_2[self.ptr: self.ptr + q_size] = feats_2
            self.ptr += q_size