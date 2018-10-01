#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump-model-params.py

import numpy as np
import six
import argparse
import os, sys
import tensorflow as tf

from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name


'''
(hyunjoon)
  For now I am using a kind of dirty hack to get around PSROI layer issue.
  This code would not work properly except for the pva100 network.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    # parser.add_argument('--meta', help='metagraph file', required=True)
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    parser.add_argument('--dump-landmark', action='store_true', help='whether to export landmark')
    args = parser.parse_args()

    # this script does not need GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # tf.train.import_meta_graph(args.meta, clear_devices=True)

    # loading...
    if args.input.endswith('.npz'):
        dic = np.load(args.input)
    else:
        dic = varmanip.load_chkpt_vars(args.input)
    import ipdb
    ipdb.set_trace()
    dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}

    dic_to_dump = {}
    postfixes = ['W:0', 'b:0', 'beta:0', 'gamma:0', 'EMA:0']
    for k, v in dic.items():
        if not args.dump_landmark and 'landmark' in k:
            continue
        found = False
        for p in postfixes:
            if p in k:
                found = True
                break
        if found:
            dic_to_dump[k] = v
            # print(k)
    varmanip.save_chkpt_vars(dic_to_dump, args.output)
    #
    # import ipdb
    # ipdb.set_trace()
    # # varmanip.save_chkpt_vars(dic, args.output)
    #
    # # save variables that are GLOBAL, and either TRAINABLE or MODEL
    # var_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # var_model = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # var_to_dump = var_trainable + var_model
    # assert len(set(var_to_dump)) == len(var_to_dump), "TRAINABLE and MODEL variables have duplication!"
    # globvarname = [k.name for k in tf.global_variables()]
    # var_to_dump = set([k.name for k in var_to_dump if k.name in globvarname])
    #
    # for name in var_to_dump:
    #     assert name in dic, "Variable {} not found in the model!".format(name)
    #
    # dic_to_dump = {k: v for k, v in six.iteritems(dic) if k in var_to_dump}
    # varmanip.save_chkpt_vars(dic_to_dump, args.output)
