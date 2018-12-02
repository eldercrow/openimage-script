# -*- coding: utf-8 -*-
# File: basemodel.py

import argparse
import cv2
import os
import numpy as np
import tensorflow as tf

from collections import deque

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
    Conv2D, Deconv2D, MaxPooling, Dropout, BatchNorm, BNReLU, LinearWrap, GlobalAvgPooling)

# from .basemodel import (
#     maybe_freeze_affine, maybe_reverse_pad, maybe_syncbn_scope, get_bn)

# from config import config as cfg

from imagenet_utils import (
    get_imagenet_dataflow, GoogleNetResize, eval_on_ILSVRC12)
from imagenet_utils import ImageNetModel as _ImageNetModel

from openimage_utils import get_openimage_dataflow
from openimage_utils import OpenImageModel as _OpenImageModel

# TOTAL_BATCH_SIZE = 512


@contextmanager
def pvanet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
            argscope([Conv2D, FullyConnected], use_bias=False):
        yield


@layer_register(log_shape=True)
def DropBlock(x, keep_prob=None, block_size=5, drop_mult=1.0, data_format='NHWC'):
    '''
    DropBlock
    '''
    if keep_prob is None or not get_current_tower_context().is_training:
        return x

    drop_prob = (1.0 - keep_prob) * drop_mult

    assert data_format in ('NHWC', 'channels_last')
    feat_size = x.get_shape().as_list()[1]
    N = tf.to_float(tf.size(x))

    f2 = feat_size * feat_size
    b2 = block_size * block_size
    r = (feat_size - block_size + 1)
    r2 = r*r
    gamma = drop_prob * f2 / b2 / r2
    k = [1, block_size, block_size, 1]

    mask = tf.less(tf.random_uniform(tf.shape(x), 0, 1), gamma)
    mask = tf.cast(mask, dtype=x.dtype)
    mask = tf.nn.max_pool(mask, ksize=k, strides=[1, 1, 1, 1], padding='SAME')
    drop_mask = (1. - mask)
    drop_mask *= (N / tf.reduce_sum(drop_mask))
    drop_mask = tf.stop_gradient(drop_mask, name='drop_mask')
    return tf.multiply(x, drop_mask, name='dropped')


@layer_register(log_shape=True)
def inception(data, ch, ch3, kernel=3, stride=1, residual=True):
    '''
    '''
    ch1 = ch - ch3

    l0 = Conv2D('conv1', data, ch, stride, strides=stride, activation=BNReLU)
    # split to l1 and l3
    l1, l3 = tf.split(l0, [ch1, ch3], axis=-1)
    # 3x3
    l3 = Conv2D('conv3', l3, ch3, kernel, padding='SAME', activation=BNReLU)
    # concat
    lc = tf.concat([l1, l3], axis=-1)
    out = Conv2D('convc', lc, ch, 1, activation=BNReLU)
    # residual
    if residual:
        out = tf.add(data, out, name='out')
    else:
        out = tf.add(l0, out, name='out')
    return out


def _get_logits(image, num_classes=1000):

    def _get_cch(ch, mul):
        return max(16, int(np.round(np.sqrt(ch) * mul / 4.0)) * 4)

    ch_mul = 2.0

    with pvanet_argscope():
        # dropblock
        if get_current_tower_context().is_training:
            keep_prob = tf.get_variable('keep_prob', (),
                                        dtype=tf.float32,
                                        trainable=False)
        else:
            keep_prob = None

        l = image #tf.transpose(image, perm=[0, 2, 3, 1])
        # space to depth
        l = tf.space_to_depth(l, 2)
        # conv1
        l = Conv2D('conv1', l, 36, 4, strides=2, activation=BNReLU, padding='SAME')
        # conv2
        l = inception('inc0', l, 36, _get_cch(36, ch_mul))

        ch_all = [72, 144, 288]
        ch3_all = [_get_cch(c, ch_mul) for c in ch_all]
        iters = [2, 6, 4]
        for ii, (ch, ch3, it) in enumerate(zip(ch_all, ch3_all, iters)):
            for jj in range(it):
                name = 'inc{}/{}'.format(ii+1, jj+1)
                k = 3 if (jj % 2 == 0) else 5
                if jj == 0:
                    l = inception(name, l, ch, ch3, k, 2, False)
                else:
                    l = inception(name, l, ch, ch3, k, 1, True)
                if ii in (0, 1):
                    bs = 7 if ii == 0 else 5
                    l = DropBlock('{}/drop'.format(name, jj), l, keep_prob, block_size=bs)

        # # should be 7x7 at this stage, with input size (224, 224)
        # l = Conv2D('convf', l, 576, 1, activation=BNReLU)
        # s = l.get_shape().as_list()
        # l = tf.reshape(l, [-1, s[1]*s[2]*s[3]])
        # ll = tf.split(l, s[1]*s[2], -1)
        # ll = [FullyConnected('psroi_proj{}'.format(i), l, 20, activation=BNReLU) \
        #         for i, l in enumerate(ll)]
        # fc = tf.concat(ll, axis=-1)
        #
        # # fc layers
        # fc = FullyConnected('fc6/L', fc, 128, activation=None)
        # fc = FullyConnected('fc6/U', fc, 4096, activation=BNReLU)
        # # fc = Dropout('fc6/Drop', fc, rate=0.25)
        # fc = FullyConnected('fc7/L', fc, 128, activation=None)
        # fc = FullyConnected('fc7/U', fc, 4096, activation=BNReLU)
        # # fc = Dropout('fc7/Drop', fc, rate=0.25)
        #
        # logits = FullyConnected('linear', fc, num_classes, use_bias=True)

        # The original implementation
        l = Conv2D('convf', l, 1280, 1, activation=BNReLU)
        l = DropBlock('dropf', l, keep_prob, block_size=3)
        l = GlobalAvgPooling('poolf', l)

        fc = tf.layers.flatten(l)
        # fc = Dropout('dropf', fc, rate=0.1)
        logits = FullyConnected('linear', fc, num_classes, use_bias=True)
    return logits


class ImageNetModel(_ImageNetModel):
    weight_decay = 4e-5
    data_format = 'NHWC'

    def get_logits(self, image, num_classes=1000):
        return _get_logits(image, num_classes)


class OpenImageModel(_OpenImageModel):
    weight_decay = 4e-5
    data_format = 'NHWC'

    def get_logits(self, image, num_classes=1502):
        return _get_logits(image, num_classes)


def get_data(name, batch, binary=False, parallel=6):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            # use lighter augs if model is too small
            GoogleNetResize(crop_area_fraction=0.16),
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

    df = get_imagenet_dataflow(args.data, name, batch, augmentors, binary, parallel) \
            if args.dataset == 'imagenet' else \
            get_openimage_dataflow(args.data, name, batch, augmentors)
    return df


def get_config(model, nr_tower):
    batch = args.batch
    parallel = args.parallel
    binary = args.binary

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch, binary, parallel)
    dataset_val = get_data('val', batch, binary, parallel)

    num_example = 1280000 if args.dataset == 'imagenet' else 1592085
    step_size = num_example // (batch * nr_tower)
    max_iter = int(step_size * 240)
    max_epoch = int(np.ceil(max_iter // step_size))
    lr_decay = np.exp(np.log(0.001) / max_epoch)
    kp_decay = np.exp(np.log(0.95) / max_epoch)
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, args.lr),]),
        HyperParamSetterWithFunc('learning_rate',
                                 lambda e, x: x * lr_decay if e > 0 else x),
        ScheduledHyperParamSetter('keep_prob',
                                  [(0, 1.0),]),
        HyperParamSetterWithFunc('keep_prob',
                                 lambda e, x: x * kp_decay if e > 0 else x),
    ]
    # callbacks = [
    #     ModelSaver(),
    #     ScheduledHyperParamSetter('learning_rate',
    #                               [(0, 0.2), (max_iter, 0)],
    #                               interp='linear', step_based=True),
    # ]
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

    TrainCfg = TrainConfig if not args.resume else AutoResumeTrainConfig
    return TrainCfg(
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
    parser.add_argument('--batch', help='batch size', type=int, default=128)
    parser.add_argument('--lr', help='initial learning rage', type=float, default=0.25)
    parser.add_argument('--parallel', help='number of cpu workers prefetching data', type=int, default=0)
    parser.add_argument('--logdir', help='checkpoint directory', type=str, default='')
    parser.add_argument('--binary', help='use lmdb instead of raw files.', action='store_true')
    # parser.add_argument('-r', '--ratio', type=float, default=0.5, choices=[1., 0.5])
    # parser.add_argument('--group', type=int, default=8, choices=[3, 4, 8],
    #                     help="Number of groups for ShuffleNetV1")
    # parser.add_argument('--v2', action='store_true', help='Use ShuffleNetV2')
    parser.add_argument('--load', help='path to load a model from')
    parser.add_argument('--resume', action='store_true', help='resume training.')
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
        name = 'avanetv3' + '_{}'.format(args.dataset)
        if args.logdir:
            name = args.logdir
        logger.set_logger_dir(os.path.join('train_log', name))

        nr_tower = max(get_num_gpu(), 1)
        config = get_config(model, nr_tower)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))

