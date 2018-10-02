# -*- coding: utf-8 -*-
# File: basemodel.py

import argparse
import cv2
import os
import numpy as np
import tensorflow as tf

from contextlib import contextmanager

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.argscope import argscope #, get_arg_scope
# from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import logger
from tensorpack.models import (
    Conv2D, Deconv2D, MaxPooling, BatchNorm, BNReLU, LinearWrap, GlobalAvgPooling)

# from .basemodel import (
#     maybe_freeze_affine, maybe_reverse_pad, maybe_syncbn_scope, get_bn)

# from config import config as cfg

from imagenet_utils import (
    get_imagenet_dataflow, GoogleNetResize, eval_on_ILSVRC12)
from imagenet_utils import ImageNetModel as _ImageNetModel

from openimage_utils import get_openimage_dataflow
from openimage_utils import OpenImageModel as _OpenImageModel

TOTAL_BATCH_SIZE = 512


@contextmanager
def ssdnet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
            argscope([Conv2D, FullyConnected], use_bias=False):
        yield


@layer_register(log_shape=True)
def DWConv(x, kernel, padding='SAME', stride=1, data_format='NHWC', w_init=None): #, active=True):
    '''
    Depthwise conv + BN + (optional) ReLU.
    We do not use channel multiplier here (fixed as 1).
    '''
    assert data_format in ('NHWC', 'channels_last')
    channel = tf.shape(x)[-1]
    if not isinstance(kernel, (list, tuple)):
        kernel = [kernel, kernel]
    filter_shape = [1, kernel[0], kernel[1], channel, 1]

    if w_init is None:
        w_init = tf.variance_scaling_initializer(2.0)
    W = tf.get_variable('W', filter_shape, initializer=w_init)
    out = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format=data_format)

    out = BNReLU(out) #if active else BatchNorm(out)
    return out


@layer_register(log_shape=True)
def LinearBottleneck(x, och, kernel,
                     padding='SAME',
                     stride=1,
                     active=False,
                     mu=6,
                     data_format='NHWC',
                     w_init=None):
    '''
    mobilenetv2 linear bottlenet.
    '''
    assert data_format in ('NHWC', 'channels_last')

    ich = tf.shape(x)[-1]
    out = Conv2D('conv_e', x, ich*mu, activation=BNReLU)
    out = DWConv('conv_d', out, kernel, padding, stride, data_format, w_init)
    out = Conv2D('conv_p', out, och, activation=(BNReLU if active else BatchNorm))
    return out


@layer_register(log_shape=True)
def DownsampleBottleneck(x, och, kernel,
                         padding='SAME',
                         stride=2,
                         active=False,
                         mu=3,
                         data_format='NHWC',
                         w_init=None):
    '''
    mobilenetv2 linear bottlenet.
    '''
    assert data_format in ('NHWC', 'channels_last')

    ich = tf.shape(x)[-1]
    out_e = Conv2D('conv_e', x, ich*mu, activation=BNReLU)
    out_d = DWConv('conv_d', out_e, kernel, padding, stride, data_format, w_init)
    out_m = MaxPooling('pool_d', out_e, kernel, stride, padding, data_format)
    out = tf.concat([out_d, out_m])
    out = Conv2D('conv_p', out, och, activation=(BNReLU if active else BatchNorm))
    return out


@layer_register(log_shape=True)
def inception(x, och, stride, mu=6, data_format='NHWC', w_init=None):
    '''
    ssdnet inception layer.
    '''
    assert data_format in ('NHWC', 'channels_last')

    ich = tf.shape(x)[-1]
    out = Conv2D('conv_c', x, ich, activation=BatchNorm)

    o1 = LinearBottleneck('conv1', out, ich, 3, mu=mu, data_format=data_format, w_init=w_init) \
         if stride == 1 else \
         DownsampleBottleneck('conv1', out, ich, 3, mu=mu//2, data_format=data_format, w_init=w_init)
    o2 = LinearBottleneck('conv2', o1, ich//2, 3, mu=mu, data_format=data_format, w_init=w_init)
    o3 = LinearBottleneck('conv3', o2, ich//2, 5, mu=mu, data_format=data_format, w_init=w_init)

    out = tf.concat([o1, o2, o3], -1)
    if stride == 1:
        out = out + x
    return out


def _get_logits(image, num_classes=1000):
    with pvanet_argscope():
        l = image #tf.transpose(image, perm=[0, 2, 3, 1])
        # conv1
        l = Conv2D('conv1', l, 18, 4, strides=2, activation=None, padding='SAME')
        with tf.variable_scope('conv1'):
            l = BNReLU(tf.concat([l, -l], -1))
        # pool2
        l = MaxPooling('pool2', l, 3, 2, 'SAME', data_format)
        # conv3
        l = LinearBottleneck('conv3', 24, 3, mu=4, data_format=data_format)

        # inception layers
        ichs = [24, 48, 96]
        ochs = [i*2 for i in ichs]
        iters = [2, 3, 3]
        mu = 6
        for ii, (ich, och, it) in enumerate(zip(ichs, ochs, iters)):
            with tf.variable_scope('inc{}'.format(ii)):
                for jj in range(it):
                    stride = 2 if jj == 0 else 1
                    l = inception('{}'.format(jj), och, stride, mu, data_format)
        l = Conv2D('convf', l, 576, activation=BNReLU)

        # should be 7x7 from this stage, with input size (224, 224)
        s = tf.shape(l)
        l = tf.reshape(l, [s[0], -1])
        ll = tf.split(l, s[1]*s[2], -1)
        ll = [FullyConnected('psroi_proj{}'.format(i), l, 20, activation=BNReLU) \
                for i, l in enumerate(ll)]
        fc = tf.concat(ll, axis=-1)

        # fc layers
        fc = FullyConnected('fc6/L', fc, 128, activation=None)
        fc = FullyConnected('fc6/U', fc, 4096, activation=BNReLU)
        fc = Dropout('fc6/Drop', fc, rate=0.25)
        fc = FullyConnected('fc7/L', fc, 128, activation=None)
        fc = FullyConnected('fc7/U', fc, 4096, activation=BNReLU)
        fc = Dropout('fc7/Drop', fc, rate=0.25)

        logits = FullyConnected('linear', fc, num_classes, use_bias=True)
        # l = GlobalAvgPooling('poolf', l, data_format)
        #
        # fc = tf.layers.flatten(l)
        # logits = FullyConnected('linear', fc, num_classes, use_bias=True)
    return logits


class ImageNetModel(_ImageNetModel):
    weight_decay = 4e-5
    data_format = 'NHWC'

    def get_logits(self, image, num_classes=1000):
        return _get_logits(image, num_classes)


class OpenImageModel(_OpenImageModel):
    weight_decay = 4e-5
    data_format = 'NHWC'

    def get_logits(self, image, num_classes=601):
        return _get_logits(image, num_classes)


def get_data(name, batch):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            # use lighter augs if model is too small
            GoogleNetResize(crop_area_fraction=0.49),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]

    df = get_imagenet_dataflow(args.data, name, batch, augmentors) \
            if args.dataset == 'imagenet' else \
            get_openimage_dataflow(args.data, name, batch, augmentors)
    return df


def get_config(model, nr_tower):
    batch = TOTAL_BATCH_SIZE // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)

    num_example = 1280000 if args.dataset == 'imagenet' else 1592088
    step_size = num_example // TOTAL_BATCH_SIZE
    max_iter = 3 * 10**5
    max_epoch = (max_iter // step_size) + 1
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, 0.1), (max_iter, 0)],
                                  interp='linear', step_based=True),
    ]
    if args.dataset == 'imagenet':
        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
    else:
        infs = [ClassificationError('wrong-best1', 'val-error-best1')]

    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    # FOR DEBUG
    # TODO: remove this
    # step_size = 10

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=step_size,
        max_epoch=max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC or OpenImage dataset dir')
    # parser.add_argument('-r', '--ratio', type=float, default=0.5, choices=[1., 0.5])
    # parser.add_argument('--group', type=int, default=8, choices=[3, 4, 8],
    #                     help="Number of groups for ShuffleNetV1")
    # parser.add_argument('--v2', action='store_true', help='Use ShuffleNetV2')
    parser.add_argument('--load', help='path to load a model from')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'openimage'],
                        help='dataset type, can be either imagenet or openimage')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # if args.v2 and args.group != parser.get_default('group'):
    #     logger.error("group= is not used in ShuffleNetV2!")

    model = ImageNetModel() if args.dataset == 'imagenet' else OpenImageModel()

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    elif args.flops:
        assert args.dataset == 'imagenet', 'Use dataset=imagenet for flop computation'
        # manually build the graph with batch=1
        input_desc = [
            InputDesc(tf.float32, [1, 224, 224, 3], 'input'),
            InputDesc(tf.int32, [1], 'label')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=False):
            model.build_graph(*input.get_input_tensors())
        model_utils.describe_trainable_vars()

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
        logger.info("Note that TensorFlow counts flops in a different way from the paper.")
        logger.info("TensorFlow counts multiply+add as two flops, however the paper counts them "
                    "as 1 flop because it can be executed in one instruction.")
    else:
        name = 'PVANETv10.0'
        logger.set_logger_dir(os.path.join('train_log', name))

        nr_tower = max(get_num_gpu(), 1)
        config = get_config(model, nr_tower)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
