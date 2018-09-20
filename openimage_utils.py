# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData, MapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger

from openimage import OpenImage, OpenImageFiles, OpenImageMeta


def get_openimage_dataflow(
        datadir, name, batch_size,
        augmentors, parallel=None):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if isTrain:
        ds = OpenImage(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = OpenImageFiles(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, labels, weights = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, labels, weights
        # ds = MapData(ds, mapf)
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_OpenImage(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'labels', 'weights'],
        output_names=['wrong-best1',]
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1 = RatioCounter()
    for best1 in pred.get_result():
        batch_size = best1.shape[0]
        acc1.feed(best1.sum(), batch_size)
    print("Best1 Error: {}".format(acc1.ratio))


class OpenImageModel(ModelDesc):
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    weight_decay = 1e-4

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    weight_decay_pattern = '.*/W'

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    def inputs(self):
        return [tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                tf.placeholder(tf.float32, [None, 601], 'labels'),
                tf.placeholder(tf.float32, [None, 601], 'weights')]

    def build_graph(self, image, labels, weights):
        image = OpenImageModel.image_preprocess(image, bgr=self.image_bgr)
        assert self.data_format in ['NCHW', 'NHWC']
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        sigmoid_loss, softmax_loss = OpenImageModel.compute_loss_and_error(logits, labels, weights)

        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary([sigmoid_loss, softmax_loss, wd_loss])
            total_cost = tf.add_n([sigmoid_loss, softmax_loss, wd_loss], name='cost')
        else:
            total_cost = tf.add_n([sigmoid_loss, softmax_loss], name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image, num_classes):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits, before softmax and/or sigmoid
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    @staticmethod
    def image_preprocess(image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, labels, weights):
        # sigmoid cross entropy loss
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                               logits=logits,
                                                               name='sigmoid_ce')
        sigmoid_loss = tf.reduce_mean(sigmoid_loss * weights, name='sigmoid_loss')

        softmax = tf.nn.softmax(logits, axis=-1)
        # reweight softmax
        w_softmax = softmax * tf.to_float(weights)
        w_softmax = tf.truediv(w_softmax, tf.maximum(1.0, tf.reduce_sum(w_softmax, axis=-1, keepdims=True)))
        # loss = -log(sum of probs from all positive labels)
        sum_w_softmax = tf.maximum(1e-08,
                                   tf.reduce_sum(w_softmax * tf.to_float(labels), axis=-1, keepdims=True))
        softmax_loss = tf.reduce_mean(-tf.log(sum_w_softmax), name='softmax_loss')
        # softmax_loss = tf.truediv(softmax_loss, 10.0, name='scaled_softmax_loss')

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        # loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_best1(logits, labels, weights, name):
            with tf.name_scope('prediction_best1'):
                _, top1 = tf.nn.top_k(logits, k=1)
                top1 = tf.one_hot(top1[:, 0], depth=tf.shape(logits)[1])
                x = tf.reduce_max(top1 * tf.to_float(labels), axis=1)
            return tf.subtract(1.0, tf.to_float(x), name=name)
            # return tf.cast(x, tf.float32, name=name)

        best1 = prediction_best1(logits, labels, weights, name='wrong-best1')
        add_moving_summary(tf.reduce_mean(best1, name='train-error-best1'))

        return sigmoid_loss, softmax_loss


if __name__ == '__main__':
    import argparse
    from tensorpack.dataflow import TestDataSpeed
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--aug', choices=['train', 'val'], default='val')
    args = parser.parse_args()

    if args.aug == 'val':
        augs = fbresnet_augmentor(False)
    elif args.aug == 'train':
        augs = fbresnet_augmentor(True)
    df = get_imagenet_dataflow(
        args.data, 'train', args.batch, augs)
    # For val augmentor, Should get >100 it/s (i.e. 3k im/s) here on a decent E5 server.
    TestDataSpeed(df).start()
