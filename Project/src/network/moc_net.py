from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from .branch import MOC_Branch
from .dla import MOC_DLA
from .resnet import MOC_ResNet

backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}


class MOC_Net(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K, flip_test=False):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.backbone = backbone[arch](num_layers)
        self.branch = MOC_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)

    def forward(self, input, textdata):
        if self.flip_test:
            assert (self.K == len(input) // 2)
            chunk1 = []
            text1 = []
            chunk2 = []
            text2 = []

            for i in range(self.K):
                chunk1_data, text1_data = self.backbone(input[i], textdata[i])
                chunk1.append(chunk1_data)
                text1.append(text1_data)

                chunk2_data, text2_data = self.backbone(input[i + self.K], textdata[i + self.K])
                chunk2.append(chunk2_data)
                text2.append(text2_data)

            return [self.branch(chunk1, text1), self.branch(chunk2, text2)]
        else:
            # print('input_chunk:', input[0].size()) # input_chunk: torch.Size([8, 3, 288, 288])

            chunk_list = []
            text_data_list = []
            for i in range(self.K):
                chunk_data, text_data = self.backbone(input[i], textdata[i])
                chunk_list.append(chunk_data)
                text_data_list.append(text_data)

            return [self.branch(chunk_list, text_data_list)]

    # def forward(self, input):
    #     if self.flip_test:
    #         assert(self.K == len(input) // 2)
    #         chunk1 = [self.backbone(input[i]) for i in range(self.K)]
    #         chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]
    #
    #         return [self.branch(chunk1), self.branch(chunk2)]
    #     else:
    #         chunk = [self.backbone(input[i]) for i in range(self.K)]
    #
    #         return [self.branch(chunk)]
