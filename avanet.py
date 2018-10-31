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
def pvanet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
            argscope([Conv2D, FullyConnected], use_bias=False):
        yield


@layer_register(log_shape=True)
def inception(data, ch, t, ndiv):
    '''
    '''
    ich = int(ch * t)
    bch = ich // ndiv
    ch1 = int(bch * (ndiv - 1))
    ch3 = bch
    # ch2 = ch3 // 2

    l1 = Conv2D('conv1', data, ch1, 1, activation=BNReLU)
    l3 = Conv2D('conv3/1', data, ch3, 1, padding='SAME', activation=BNReLU)
    l3 = Conv2D('conv3/2', l3, ch3, 3, padding='SAME', activation=BNReLU)

    lc = tf.concat([l1, l3], axis=-1)
    lc = Conv2D('convc', lc, ch, 1, activation=None)
    with tf.variable_scope('convc'):
        lc = BatchNorm('bn', lc)
    return tf.add(data, lc, name='out')

@layer_register(log_shape=True)
def downception(data, ch, t, ndiv):
    '''
    '''
    ich = int(ch * t)
    bch = ich // ndiv
    ch1 = int(bch * (ndiv - 1))
    ch3 = bch
    ch2 = ch3 // 2

    l1 = Conv2D('conv1', data, ch1, 1, activation=BNReLU)
    l1 = MaxPooling('pool1', l1, 2, strides=2, padding='SAME')
    l3 = Conv2D('conv3/1', data, ch2, 1, padding='SAME', activation=BNReLU)
    l3 = Conv2D('conv3/2', l3, ch3, 4, strides=2, padding='SAME', activation=BNReLU)

    lc = tf.concat([l1, l3], axis=-1)
    lc = Conv2D('convc', lc, ch, 1, activation=None)
    with tf.variable_scope('convc'):
        lc = BatchNorm('bn', lc)
    return lc


def _get_logits(image, num_classes=1000):
    with pvanet_argscope():
        l = image #tf.transpose(image, perm=[0, 2, 3, 1])
        # conv1
        l = Conv2D('conv1', l, 12, 4, strides=2, activation=None, padding='SAME')
        with tf.variable_scope('conv1'):
            l = BNReLU(tf.concat([l, -l], -1))
        # conv2
        l = Conv2D('conv2', l, 48, 1, activation=BNReLU)
        l = MaxPooling('pool2', l, 2, strides=2, padding='SAME')
        # conv3
        l = Conv2D('conv3', l, 24, 1, activation=None)
        l = BatchNorm('conv3/bn', l)

        channels = [48, 96, 192]
        iters = [2, 6, 3]
        ndivs = [4, 6, 6]
        mults = [2, 3, 3]
        for ii, (ch, it, ndiv, t) in enumerate(zip(channels, iters, ndivs, mults)):
            for jj in range(it):
                name = 'inc{}/{}'.format(ii+1, jj+1)
                if jj == 0:
                    l = downception(name, l, ch, t, ndiv)
                else:
                    l = inception(name, l, ch, t, ndiv)

        # should be 7x7 from this stage, with input size (224, 224)
        s = l.get_shape().as_list()
        l = tf.reshape(l, [-1, s[1]*s[2]*s[3]])
        ll = tf.split(l, s[1]*s[2], -1)
        ll = [FullyConnected('psroi_proj{}'.format(i), l, 20, activation=BNReLU) \
                for i, l in enumerate(ll)]
        fc = tf.concat(ll, axis=-1)
        # l = Conv2D('psroi_proj', l, 20, 1)

        fc = tf.layers.flatten(l)

        # fc layers
        fc = FullyConnected('fc6/L', fc, 128, activation=None)
        fc = FullyConnected('fc6/U', fc, 4096, activation=BNReLU)
        fc = Dropout('fc6/Drop', fc, rate=0.25)
        fc = FullyConnected('fc7/L', fc, 128, activation=None)
        fc = FullyConnected('fc7/U', fc, 4096, activation=BNReLU)
        fc = Dropout('fc7/Drop', fc, rate=0.25)

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


def get_data(name, batch):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            # use lighter augs if model is too small
            GoogleNetResize(crop_area_fraction=0.36),
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

    num_example = 1280000 if args.dataset == 'imagenet' else 1592085
    step_size = num_example // TOTAL_BATCH_SIZE
    max_iter = 6 * 10**5
    max_epoch = (max_iter // step_size) + 1
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, 0.2), (max_iter, 0)],
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
        name = 'PVANETv11.0'
        logger.set_logger_dir(os.path.join('train_log', name))

        nr_tower = max(get_num_gpu(), 1)
        config = get_config(model, nr_tower)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
