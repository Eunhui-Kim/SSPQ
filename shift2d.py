# -*- coding: utf-8 -*-
# File: shift2d.py
# Author : Eunhui Kim

from ..compat import tfv1 as tf  # this should be avoided first in model code

from ..tfutils.common import get_tf_version_tuple
from ..utils.argtools import get_data_format, shape2d, shape4d, log_once
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args, rename_get_variable
from tensorflow_active_shift.python.ops import active_shift2d_ops
import numpy as np

__all__ = ['Shift2D']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['n_shift','shift_size'],
    name_mapping={
        'shift_shape': 'n_shift',
        'out_channel': 'shift_size',
        'strides': 'stride',
        'paddings' : 'padding'
    })
def Shift2D(
        inputs,
        n_shift,     #number of shifts position to compute
        shift_size,  #number of channels
        stride=1,
        padding='SAME', #'same',
        data_format='NCHW'):
        #dilation_rate=(1, 1),
        #kernel_initializer=None,
        #kernel_regularizer=None,
        #bias_regularizer=None,
        #split=1):
    """
    It's for 'tensorflow.active_shift_2d' into tensor graph to compute the gradient of shifts

    1. Default shift initializer is np_random_uniform(-1, 1, (2, in_channels)) .
    2. set strides and paddings as a list of integer by the definition
 
    Variable Names:

    * ``SP``: shifts positions (row, col) for # of in_channels
    """
    
    #shift_tensor = np.random.uniform(low=-1, high=1, size=(filters, kernel_size)) #(2, num_filters_in)

    if stride == 1:
    	strides = [1, 1, 1, 1]
    elif stride == 2:
        strides = [1, 1, 2, 2]
    elif stride == 3:
        strides = [1, 1, 3, 3]

    h_rate = 1
    w_rate = 1

    if padding=='SAME':
      paddings = [0, 0, 0, 0]
    elif padding=='AUTO':
      if stride == 1:
        paddings = [0, 0, 0, 0]
      else:
        kernel_height_effective = 3
        kernel_width_effective = 3
        kernel_height_effective = kernel_height_effective + (kernel_height_effective - 1) * (h_rate - 1)
        kernel_width_effective = kernel_width_effective + (kernel_width_effective - 1) * (w_rate - 1)
        pad_h_beg = (kernel_height_effective - 1) // 2
        pad_h_end = kernel_height_effective - 1 - pad_h_beg
        pad_w_beg = (kernel_width_effective - 1) // 2
        pad_w_end = kernel_width_effective - 1 - pad_w_beg
        padding = [[0, 0], [pad_h_beg, pad_h_end], [pad_w_beg, pad_w_end], [0, 0]]
        if data_format == 'NCHW':
		       padding = [padding[0], padding[3], padding[1], padding[2]]
         #inputs = tf.pad(inputs, padding)
        paddings = [0, 0, pad_h_beg, pad_w_beg]

    #init = shift_tensor
    filter_shape = (n_shift, shift_size)
    input_dtype = inputs.dtype
    with tf.variable_scope('SP', tf.AUTO_REUSE):
            SP = tf.get_variable('SP', filter_shape, dtype=input_dtype, initializer = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform'))
            layer = active_shift2d_ops.active_shift2d_op(
                inputs,
                shift=SP,
                strides=strides,
                paddings=paddings,
                data_format=data_format) #,
                #kernel_regularizer=kernel_regularizer,
                #activity_regularizer=activity_regularizer,
                #_reuse=tf.get_variable_scope().reuse)
            #ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(layer, name='output')

    ret.variables = VariableHolder(SP=SP)
    return ret

