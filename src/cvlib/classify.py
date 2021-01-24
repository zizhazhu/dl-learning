import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    # 构建一层cnn，完成的模型层
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding = 1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size = 2, stride = 2))
    # 原作者在 paper 裡是說她在 omniglot 用的是 strided convolution
    # 不過這裡我改成 max pool (mini imagenet 才是 max pool)
    # 這並不是你們在 report 第三題要找的 tip


class Classifier(nn.Module):
    # 普通cnn模型，输入图像，输出logits

    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()
        self.conv1 = conv_block(in_ch, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        '''
        for block in range(1, 5):
            x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], params[f'conv{block}.0.bias'],
                                  params.get(f'conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))

        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x