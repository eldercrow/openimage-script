## Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""SSDNet implemented in Gluon."""
__all__ = ['SSDNet', 'ssdnet1_0', 'get_ssdnet']

__modify__ = 'dwSun'
__modified_date__ = '18/04/18'

import os

from mxnet.gluon import nn
from mxnet.context import cpu
# from mxnet.gluon.block import HybridBlock
from mxnet import base


class ConvBNReLU(nn.HybridBlock):
    '''
    conv-bn-relu block.
    '''
    def __init__(self, num_filter, kernel,
            stride=1, pad=-1, num_group=1, active=True, crelu=False, **kwargs):
        #
        super(ConvBNReLU, self).__init__(**kwargs)
        #
        if pad < 0:
            pad = (kernel - 1) // 2

        self.crelu = crelu
        self.active = active

        with self.name_scope():
            # conv
            conv = nn.HybridSequential()
            conv.add(nn.Conv2D(num_filter, kernel, stride, pad, groups=num_group, use_bias=False))
            self.conv = conv
            # bn
            bn = nn.HybridSequential()
            bn.add(nn.BatchNorm(scale=True))
            self.bn = bn
            # relu
            if active:
                relu = nn.HybridSequential()
                relu.add(nn.Activation('relu'))
                self.relu = relu

    def hybrid_forward(self, F, x):
        #
        out = self.conv(x)
        if self.crelu:
            out = F.concat(out, -out)
        out = self.bn(out)
        if self.active:
            out = self.relu(out)
        return out


class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    """

    def __init__(self, ich, och, kernel, t, pad=-1, active=True, **kwargs):
        #
        super(LinearBottleneck, self).__init__(**kwargs)
        #
        with self.name_scope():
            out = nn.HybridSequential()
            out.add(ConvBNReLU(ich*t, 1, active=True))
            out.add(ConvBNReLU(ich*t, kernel, num_group=ich*t, active=active))
            out.add(ConvBNReLU(och, 1, active=False))
            self.out = out

    def hybrid_forward(self, F, x):
        out = self.out(x)
        return out


class DownsampleBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    """

    def __init__(self, ich, och, t, pad=-1, **kwargs):
        #
        super(DownsampleBottleneck, self).__init__(**kwargs)
        #
        with self.name_scope():
            self.out_e = nn.HybridSequential()
            self.out_e.add(ConvBNReLU(ich*t, 1, active=True))

            self.out_d1 = nn.HybridSequential()
            self.out_d1.add(ConvBNReLU(ich*t, 4, stride=2, num_group=ich*t, active=True))

            self.out_d2 = nn.HybridSequential()
            self.out_d2.add(ConvBNReLU(ich*t, 4, stride=2, num_group=ich*t, active=True))

            self.out = nn.HybridSequential()
            self.out.add(ConvBNReLU(och, 1, active=False))

    def hybrid_forward(self, F, x):
        out_e = self.out_e(x)
        out_d1 = self.out_d1(out_e)
        out_d2 = self.out_d2(out_e)
        out = self.out(F.concat(out_d1, out_d2))
        return out


class Inception(nn.HybridBlock):
    '''
    Basic inception block in SSDNet.
    '''
    def __init__(self, ich, och, t, stride=1, swap_block=False, **kwargs):
        #
        super(Inception, self).__init__(**kwargs)
        #
        assert stride in (1, 2)
        self.use_shortcut = stride == 1 and int(ich*2) == och
        self.swap_block = swap_block

        with self.name_scope():
            o1 = nn.HybridSequential(prefix='o1')
            if stride == 1:
                o1.add(LinearBottleneck(ich, ich, 3, t))
            else:
                o1.add(DownsampleBottleneck(ich//2, ich, t))
            self.o1 = o1

            o2 = nn.HybridSequential(prefix='o2')
            o2.add(LinearBottleneck(ich//2, ich, 5, t, active=False))
            self.o2 = o2

    def hybrid_forward(self, F, x):
        #
        o1 = self.o1(x)
        o2 = self.o2(o1)
        if self.swap_block:
            out = F.concat(o2, o1)
        else:
            out = F.concat(o1, o2)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


# Net
class SSDNet(nn.HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        #
        super(SSDNet, self).__init__(**kwargs)
        #
        with self.name_scope():
            out = nn.HybridSequential(prefix='')
            with out.name_scope():
                out.add(ConvBNReLU(18, 4, stride=2, crelu=True, prefix='conv1'))

                out.add(ConvBNReLU(36, 1, crelu=True, prefix='conv2/exp'))
                out.add(ConvBNReLU(72, 4, stride=2, num_group=72, prefix='conv2/dw'))
                out.add(ConvBNReLU(24, 1, active=False, prefix='conv2'))

                ich_all = [24, 48, 96]
                och_all = [c*2 for c in ich_all]
                iter_all = [2, 4, 2]

                for ii, (ich, och, it) in enumerate(zip(ich_all, och_all, iter_all), 1):
                    for jj in range(it):
                        stride = 2 if jj == 0 else 1
                        swap_block = (jj%2) == 1
                        prefix = 'inc{}/{}'.format(ii, jj)
                        out.add(Inception(ich, och, 6, stride, swap_block, prefix=prefix))

                out.add(ConvBNReLU(1280, 1, prefix='convf'))
                out.add(nn.GlobalAvgPool2D())
                out.add(nn.Flatten())
            self.features = out

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Constructor
def get_ssdnet(multiplier, ctx=cpu(), **kwargs):
    '''
    '''
    net = SSDNet(multiplier, **kwargs)
    return net


def ssdnet1_0(**kwargs):
    '''
    '''
    return get_ssdnet(1.0, **kwargs)
