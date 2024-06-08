from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def create_fig(tensor, type):
    # 将张量展平为一维数组
    flat_tensor = tensor.cpu().flatten().detach().numpy()
    # 绘制直方图
    name = 'Tensor Element Distribution ' + type
    plt.hist(flat_tensor, bins=15, color='blue', edgecolor='black')
    plt.title(name)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    # 保存图片
    plt.savefig(name + '.png')


import torch
import torch.nn as nn
import torch.nn.functional as F


# class CrossAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(CrossAttention, self).__init__()
#
#         # 定义注意力权重计算的线性层
#         self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2):
#         x2 = x2 / 5
#         batch_size, channels, height, width = x1.size()
#
#         # 通过线性层计算查询、键、值
#         proj_query = self.query_conv(x1).view(batch_size, -1, width * height).permute(0, 2, 1)
#         proj_key = self.key_conv(x2).view(batch_size, -1, width * height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = F.softmax(energy, dim=-1)
#
#         proj_value = self.value_conv(x2).view(batch_size, -1, width * height)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#
#         out = out.view(batch_size, channels, height, width)
#         out = self.gamma * out + x1
#
#         return out

class GatingModule(nn.Module):
    def __init__(self):
        super(GatingModule, self).__init__()
        self.conv1 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_feature, text_feature):
        # 缩小feature2中的每个元素5倍
        text_feature = text_feature / 10
        # 拼接两个特征
        combined_features = torch.cat((image_feature, text_feature), dim=1)
        # 计算门控权重
        gate_weight = self.conv1(combined_features)
        gate_weight = self.conv2(gate_weight)
        gate_weight = self.conv3(gate_weight)
        gate_weight = self.sigmoid(gate_weight)

        # 使用门控权重融合两个特征
        fused_feature = image_feature * (1 - gate_weight) + text_feature * gate_weight
        return fused_feature


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv
        self.GatingModule = GatingModule()

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    def forward(self, input_chunk, textdata=None):
        output = {}
        output_wh = []
        merged_feature = []

        for image_feature, text_feature in zip(input_chunk, textdata):
            merged_feature.append(self.GatingModule(image_feature, text_feature))

        merged_feature = torch.cat(merged_feature, dim=1)

        for feature in input_chunk:
            output_wh.append(self.wh(feature))
        input_chunk = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output['hm'] = self.hm(merged_feature)
        output['mov'] = self.mov(input_chunk)
        output['wh'] = output_wh
        return output
