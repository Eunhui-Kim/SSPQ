# -*- coding: utf-8 -*-
# File: resnet_model.py
# Author : Eunhui Kim (modifier) starting from resnet_model.py in tensorpack

import tensorflow as tf

from tensorpack.models import BatchNorm, BNReLU, Conv2D, Shift2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from dorefa import get_dorefa

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.jjjjj
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)

# ----------------- ASL resnet ----------------------
def asl_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    shortcut = l
    l = BatchNorm('preact', l)
    l = activate(l)
    #l, shortcut = apply_preactivation(l, preact)
    l = Shift2D('shift_asl', l, n_shift=2, shift_size=ch_out, strides=stride, padding='SAME')
    l = Conv2D('conv1', l, ch_out * 4, kernel_size=1, strides=1, activation=get_bn(zero_init=False))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def asl_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l
# ----------------- ASL resnet ----------------------

# ----------------- ResIB-Shift resnet ----------------------

def resIBShift_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    ch_in = l.get_shape().as_list()[1]
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Shift2D('shift1', l, n_shift=2, shift_size=ch_out, strides=stride, padding='SAME')
    l = Conv2D('conv3', l, ch_out * 4, kernel_size=1, strides=1, activation=get_bn(zero_init=False))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def resIBShift_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l
# ----------------- ResIB-Shift resnet ----------------------

# ----------------- Quntized ResIB-Shift resnet ----------------------
BITW = 1
BITA = 4
BITG = 32

fw, fa, fg = get_dorefa(BITW, BITA, BITG)

def new_get_variable(v):
    name = v.op.name
    #skip binarize the first and last layer
    if not name.endwith('W') or 'conv1' in name or 'fct' in name:
        return v
    else:
        logger.info("Binarized weight {}".format(v.op.name))
        return fw(v)

def new_w(v):
    return fw(v)

def nonlin(x):
    return tf.clip_by_value(x, 0.0, 1.0)

def activate(x):
    return fa(nonlin(x))

def apply_preactivation2(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BatchNorm('preact', l)
        l = activate(l)
    else:
        shortcut = l
    return l, shortcut

def QresIBShift_bottleneck(l, ch_out, stride, preact):
    def get_stem_shift(l):
        l = Conv2D('conv1x1a', l, ch_out, kernel_size=1, strides=1)
        l = BatchNorm('stembn', l)
        l = activate(l)
        l = Shift2D('shift_q', l, n_shift=2, shift_size=ch_out, strides=stride, padding='SAME')
        l = Conv2D('conv1x1b', l, ch_out * 4, kernel_size=1, strides=1)
        l = BatchNorm('stembn2', l)
        return l

    l, shortcut = apply_preactivation2(l, preact)

    stem = get_stem_shift(l)
    return stem + resnet_shortcut(shortcut, ch_out * 4, stride)

def QresIBShift_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                    # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
            # end of each group need an extra activation
        l = BatchNorm('bnlast', l)
        l = activate(l)
    return l
# -----------------Quntized ResIB-Shift resnet ----------------------

# ----------------- pre-activation resnet ----------------------

def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut

def preact_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preact_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preact_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l
# ----------------- pre-activation resnet ----------------------


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def se_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)

def se_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l

def resnext32x4d_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out * 2, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * 2, 3, strides=stride, activation=BNReLU, split=32)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)

def resnext32x4d_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l

def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that TF pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits
